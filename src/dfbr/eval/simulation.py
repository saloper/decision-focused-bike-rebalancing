import pandas as pd
import numpy as np
import torch

#Define a station object to keep track of inventory and handle constraints
class Station:
    def __init__(self, id, capacity, closest_station, lat, long, inventory=0):
        self.id = id
        self.capacity = capacity
        self.closest_station = closest_station
        self.inventory = inventory
        self.lat = lat
        self.long = long
        #Dicts to track metrics overtime
        self.lost_demand = {}
        self.over_capacity = {}
        self.forced_returns = {}

        #Track state changes
        self.history = []

    def log_history(self, time):
        self.history.append({'time' : time, 'inventory' : self.inventory})

    def return_bike(self, time):
        if self.inventory < self.capacity:
            self.inventory +=1
            self.log_history(time)
            return True
        else:
            return False
        
    def force_return_bike(self, time):
        self.inventory += 1
        self.log_history(time)

    
    def rent_bike(self, time):
        if self.inventory > 0 :
            self.inventory -=1
            self.log_history(time)
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
        station_data = station_data[['Total Docks', 'Latitude', 'Longitude', 'Closest Station']].to_dict('index')
        
        #Add Stations to simulation
        stations = {}
        for id, data in station_data.items():
            stations[id] = Station(
                id=id,
                capacity=data['Total Docks'],
                closest_station=data['Closest Station'],
                lat = data['Latitude'],
                long = data['Longitude'],
                inventory= int(data['Total Docks'] * start_inv_pct)
            )
        return stations

#Create a dataframe of events to simmulate through
def create_event_df(trip_file_path, station_file_path, start_date, end_date, cutoff_hour = None):
    #Read Raw Files
    trips = pd.read_parquet(trip_file_path, engine='pyarrow')
    stations = pd.read_parquet(station_file_path, engine='pyarrow')

    #Filter trips to normal
    #trips = trips[trips['Closed Status'] == 'NORMAL']

    #Filer to data range
    start_thresh = pd.to_datetime(start_date).tz_localize('America/New_York')
    end_thresh = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).tz_localize('America/New_York')
    trips = trips[
        (trips['Start Date'] >= start_thresh) & 
        (trips['End Date'] < end_thresh)
    ]

    #Inner Join to filter out station id changes
    # trips['Start Station Name'] = trips['Start Station Name'].str.lower().str.replace(' and ', ' & ', regex=False)
    # trips['End Station Name'] = trips['End Station Name'].str.lower().str.replace(' and ', ' & ', regex=False)
    # stations['Name'] = stations['Name'].str.lower().str.replace(' and ', ' & ', regex=False)
    filtered = pd.merge(trips, stations, left_on=['Start Station Id'], right_on=['Id'], how='inner')
    filtered = pd.merge(filtered, stations, left_on=['End Station Id'], right_on=['Id'], how='inner')
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
    
    if cutoff_hour is not None:
        # Keep only events where the hour is strictly less than the cutoff (e.g., < 12 keeps 0 through 11)
        event_df = event_df[event_df['time'].dt.hour < cutoff_hour]

    event_df.sort_values(by=['time', 'event_type'], inplace=True)

    return event_df


#Define a Simulation object to orchestrate bikes and stations
class Sim:
    def __init__(self, station_dict, station_ids, event_df, num_stations, max_cap, predict_ds= None, predict_model=None, cost_head=None, opt_model=None, scaling=None):
        self.stations = station_dict
        self.event_df = event_df
        self.current_time = None
        self.predict_ds = predict_ds

        self.num_stations = num_stations
        self.max_cap = max_cap
        #Sort again to be safe
        self.station_ids = station_ids
        #Create mappings from ids and dates to indices
        self.id_to_idx = {s_id : i for i, s_id in enumerate(self.station_ids)}
        self.idx_to_id = {i : s_id for i, s_id in enumerate(self.station_ids)}
        if self.predict_ds:
            self.date_to_idx = {d : i for i, d in enumerate(self.predict_ds.dates)}
        self.predict_model = predict_model
        self.scaling = scaling
        self.cost_head = cost_head
        self.opt_model = opt_model
        #Track metrics
        self.moved_bikes = {}
        self.lost_demand = {}
        self.over_capacity = {}
        self.forced_returns = {}
        self.total_inventory = {}

        self.failed_trips = set()

    def _execute_rebalance(self, cp):
            #Run optimization model
            self.opt_model.setObj(cp.squeeze(0))
            wp, _ = self.opt_model.solve()
            #Reshape into targets
            wp = np.array(wp).reshape(self.num_stations, self.max_cap+1)
            targets = np.argmax(wp, axis =1)
            total_moves = 0
            #Apply the targets
            for i in range(len(self.station_ids)):
                station_id = self.idx_to_id[i]
                #print(f"Station: {station_id} Inventory: {self.stations[station_id].inventory} Target: {targets[i]}")
                total_moves += abs(self.stations[station_id].inventory - targets[i])
                self.stations[station_id].inventory = targets[i]
            
            return total_moves

    def run(self):
        #Group into discrete days
        daily_groups = self.event_df.groupby(self.event_df['time'].dt.date)

        #Daily loop
        for current_date, daily_events in daily_groups:
            
            #Initialize daily metrics
            #Simulation level
            self.lost_demand[current_date] = 0
            self.over_capacity[current_date] = 0
            self.forced_returns[current_date] = 0
            self.total_inventory[current_date] = sum(station.inventory for station in self.stations.values())
            
            #Station Level
            for station in self.stations.values():
                station.lost_demand[current_date] = 0
                station.over_capacity[current_date] = 0
                station.forced_returns[current_date] = 0

        #Overnight rebalancing
            if self.predict_ds and self.predict_model and self.cost_head and self.opt_model:
                idx = self.date_to_idx.get(current_date)
                if idx is not None:
                    #Get scaled features from dataset
                    x, y, _ = self.predict_ds[idx]
                    #Inference
                    self.cost_head.eval()
                    self.predict_model.eval()
                    with torch.no_grad():
                        yp_scaled = self.predict_model(x.unsqueeze(0))
                        yp_unscaled = (yp_scaled * self.scaling["y_std"]) + self.scaling["y_mean"]
                        cp = self.cost_head(yp_unscaled)
                    
                    #Move bikes based on forecast
                    self.moved_bikes[current_date] = self._execute_rebalance(cp)

                    #Log event in stations history
                    midnight = pd.to_datetime(current_date).tz_localize('America/New_York')
                    for station in self.stations.values():
                        station.log_history(midnight)

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
                    if not station.rent_bike(self.current_time):
                        station.lost_demand[current_date] += 1  
                        self.failed_trips.add(row.trip_id)

                elif row.event_type == 'return':
                    if not station.return_bike(self.current_time):
                        station.over_capacity[current_date] += 1
                        #Try again at the next closest station
                        station = self.stations.get(station.closest_station)
                        if not station.return_bike(self.current_time):
                            station.forced_returns[current_date] += 1
                            #Dump the bike
                            station.force_return_bike(self.current_time)
                
            #Summarize system metrics
            self.lost_demand[current_date] = sum(station.lost_demand[current_date] for station in self.stations.values())
            self.over_capacity[current_date] = sum(station.over_capacity[current_date] for station in self.stations.values())
            self.forced_returns[current_date] = sum(station.forced_returns[current_date] for station in self.stations.values())