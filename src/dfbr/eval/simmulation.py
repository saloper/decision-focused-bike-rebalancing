import pandas as pd
import numpy as np
import torch

#Define a station object to keep track of inventory and handle constraints
class Station:
    def __init__(self, id, capacity, closest_station, inventory=0):
        self.id = id
        self.capacity = capacity
        self.closest_station = closest_station
        self.inventory  = inventory
    
    def return_bike(self):
        if self.inventory < self.capacity:
            self.inventory +=1
            return True
        else:
            return False
        
    def force_return_bike(self):
        self.inventory += 1
    
    def rent_bike(self):
        if self.inventory > 0 :
            self.inventory -=1
            return True
        else:
            return False

#Create a dictionary of station objects
def create_station_dict(station_file_path, dist_file_path, start_inv_pct):
        #Read in files for stations
        station_data = pd.read_parquet(station_file_path, engine='pyarrow')
        distance_data = pd.read_parquet(dist_file_path, engine='pyarrow')
        #Sort by id 
        station_data.set_index('Id', inplace=True)
        station_data.sort_index(inplace=True)
        distance_data.sort_index(inplace=True)
        #Calculate Closest Station
        np.fill_diagonal(distance_data.values, np.inf)
        closest = distance_data.idxmin(axis=1)
        station_data['Closest Station'] = closest
        station_data = station_data[['Total Docks', 'Closest Station']].to_dict('index')
        
        #Add Stations to simulation
        stations = {}
        for id, data in station_data.items():
            stations[id] = Station(
                id=id,
                capacity=data['Total Docks'],
                closest_station=data['Closest Station'],
                inventory= int(data['Total Docks'] * start_inv_pct)
            )
        return stations

#Create a dataframe of events to simmulate through
def create_event_df(trip_file_path, station_file_path, start_date, end_date):
    #Read Raw Files
    trips = pd.read_parquet(trip_file_path, engine='pyarrow')
    stations = pd.read_parquet(station_file_path, engine='pyarrow')

    #Filter trips to normal
    trips = trips[trips['Closed Status'] == 'NORMAL']

    #Filer to data range
    start_thresh = pd.to_datetime(start_date).tz_localize('America/New_York')
    end_thresh = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).tz_localize('America/New_York')
    trips = trips[
        (trips['Start Date'] >= start_thresh) & 
        (trips['End Date'] < end_thresh)
    ]

    #Inner Join to filter out station id changes
    trips['Start Station Name'] = trips['Start Station Name'].str.lower().str.replace(' and ', ' & ', regex=False)
    stations['Name'] = stations['Name'].str.lower().str.replace(' and ', ' & ', regex=False)
    filtered = pd.merge(trips, stations, left_on=['Start Station Id', 'Start Station Name'], right_on=['Id', 'Name'], how='inner')
    filtered['trip_id'] = filtered.index

    #Extract Rents
    rents = filtered[['Start Date', 'Start Station Id', 'trip_id']].copy()
    rents.columns = ['time', 'station_id', 'trip_id']
    rents['event_type'] = 'rent'

    #Extract Returns
    returns = filtered[['End Date', 'End Station Id', 'trip_id']].copy()
    returns.columns = ['time', 'station_id', 'trip_id']
    returns['event_type'] = 'return'

    #Combine and sort
    event_df = pd.concat([rents, returns], ignore_index=True)
    event_df.sort_values(by=['time', 'event_type'], inplace=True)

    return event_df


#Define a Simulation object to orchestrate bikes and stations
class Sim:
    def __init__(self, station_dict, event_df, predict_ds= None, predict_model=None, opt_model=None):
        self.stations = station_dict
        self.event_df = event_df
        self.current_time = None
        self.predict_ds = predict_ds

        self.sorted_ids = sorted(self.stations.keys())
        self.id_to_idx = {s_id: i for i, s_id in enumerate(self.sorted_ids)}
        self.idx_to_id = {i: s_id for i, s_id in enumerate(self.sorted_ids)}
        if self.predict_ds:
            self.date_to_idx = {d: i for i, d in enumerate(self.predict_ds.dates)}
        self.predict_model = predict_model
        self.opt_model = opt_model
        #Track metrics
        self.lost_demand = {}
        self.over_capacity = {}
        self.forced_returns = {}
        self.total_inventory = {}

        self.failed_trips = set()

    def _execute_rebalance(self, forecast):
            """Passes current state to Gurobi and updates station objects"""
            # A. Gather current inventory into a flat numpy array (0-59)
            current_inv = np.array([self.stations[s_id].inventory for s_id in self.sorted_ids])

            # B. Run Gurobi
            # Returns: obj_val, flow_matrix, shortage_array, capacity_array
            _, flow, _, _ = self.opt_model.solve(current_inv, forecast)

            # C. Apply the moves to the physical station objects
            if flow is not None:
                for i in range(len(self.sorted_ids)):
                    for j in range(len(self.sorted_ids)):
                        if flow[i, j] > 0:
                            count = int(flow[i, j])
                            from_id = self.idx_to_id[i]
                            to_id = self.idx_to_id[j]
                            
                            self.stations[from_id].inventory -= count
                            self.stations[to_id].inventory += count
    def run(self):
        print('Starting Simulation!')
        #Group into discrete days
        daily_groups = self.event_df.groupby(self.event_df['time'].dt.date)

        #Daily loop
        for current_date, daily_events in daily_groups:
        # --- OVERNIGHT REBALANCING ---
            if self.predict_ds and self.predict_model and self.opt_model:
                idx = self.date_to_idx.get(current_date)
                if idx is not None:
                    # 1. Get scaled features from Dataset
                    X_scaled, _ = self.predict_ds[idx]
                    
                    # 2. Inference
                    self.predict_model.eval()
                    with torch.no_grad():
                        y_scaled = self.predict_model(X_scaled.unsqueeze(0))
                        # 3. Un-scale to get actual bike counts
                        forecast = (y_scaled * self.predict_ds.y_std) + self.predict_ds.y_mean
                        forecast = torch.clamp(torch.round(forecast), min=0).squeeze().numpy()
                    
                    # 4. Move bikes based on forecast!
                    self._execute_rebalance(forecast)

            #Initialize daily metrics
            self.lost_demand[current_date] = 0
            self.over_capacity[current_date] = 0
            self.forced_returns[current_date] = 0
            self.total_inventory[current_date] = sum(station.inventory for station in self.stations.values())

            #Loop through daily events
            for row in daily_events.itertuples(index=False):
                #Check if trying to remove a failed trip
                if row.event_type == 'return' and row.trip_id in self.failed_trips:
                    self.failed_trips.remove(row.trip_id)
                    continue
                
                #Set simulation time
                self.current_time = row.time
                #Get station
                station = self.stations.get(row.station_id)

                if row.event_type == 'rent':
                    if not station.rent_bike():
                        self.lost_demand[current_date] += 1  
                        self.failed_trips.add(row.trip_id)

                elif row.event_type == 'return':
                    if not station.return_bike():
                        self.over_capacity[current_date] += 1
                        #Try again at the next closest station
                        station = self.stations.get(station.closest_station)
                        if not station.return_bike():
                            self.forced_returns[current_date] += 1
                            #Dump the bike
                            station.force_return_bike()

        print('Simulation Complete!')