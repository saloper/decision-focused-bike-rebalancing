import numpy as np
import pandas as pd
from pathlib import Path
#--------------------------------------------------------------------------------------
#Helper Functions
#--------------------------------------------------------------------------------------
def add_temporal_encodings(df):
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    
    # Day of week
    day_of_week = df.index.dayofweek
    df['sin_day_of_week'] = np.sin(2 * np.pi * day_of_week / 7)
    df['cos_day_of_week'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # Month
    month = df.index.month
    df['sin_month'] = np.sin(2 * np.pi * month / 12)
    df['cos_month'] = np.cos(2 * np.pi * month / 12)
    
    return df

def calc_net_demand(data, freq='D'):
    temp = data.copy()
    #Filter out non-normal rides
    temp = temp[temp['Closed Status'] == 'NORMAL']

    #Calculate Net Demand
    outflow = temp.groupby([
        pd.Grouper(key='Start Date', freq=freq), 
        'Start Station Id', 
        'Start Station Name'
    ]).size().rename('Outflow').to_frame()

    inflow = temp.groupby([
        pd.Grouper(key='End Date', freq=freq), 
        'End Station Id', 
        'End Station Name'
    ]).size().rename('Inflow').to_frame()

    outflow.index.names = ['Date', 'Station Id', 'Station Name']
    inflow.index.names = ['Date', 'Station Id', 'Station Name']

    flow = outflow.join(inflow, how='outer').fillna(0).reset_index()
    flow['Netflow'] = flow['Inflow'] - flow['Outflow'] 

    print(f"\nSuccess! Combined flow at {freq} frequency. Total rows: {len(flow)}")
    return flow

#--------------------------------------------------------------------------------------
#Main
#--------------------------------------------------------------------------------------
if __name__ == '__main__':
    current_dir = Path(__file__).parent
    root = current_dir.parent.parent
    data_dir = root / "data" 
    
    #Read Raw Files
    trip = pd.read_csv(data_dir / "raw" / "pogoh_trip_data.csv")
    station = pd.read_csv(data_dir / "raw" / "pogoh_station_data.csv")
    
    #Inner Join to filter out station id changes
    trip['Station Name'] = trip['Station Name'].str.lower().str.replace(' and ', ' & ', regex=False)
    station['Name'] = station['Name'].str.lower().str.replace(' and ', ' & ', regex=False)
    filtered = pd.merge(trip, station, left_on=['Station Id', 'Station Name'], right_on=['Id', 'Name'], how='inner')
    filtered = filtered[['Date', 'Id', 'Netflow']]

    #Getlist of all IDs
    station_ids = sorted(station['Id'].unique())
    
    #Pivot the data
    filtered = filtered.pivot_table(index='Date', columns='Id', values='Netflow', aggfunc='sum', fill_value=0)
    
    #Force the columns to match station list, filling missing ones with 0
    filtered = filtered.reindex(columns=station_ids, fill_value=0)
    filtered = add_temporal_encodings(filtered)
    
    #Split train vs. test
    train = filtered.loc[filtered.index < '2025-02-01']
    test = filtered.loc[filtered.index >= '2025-02-01']
    
    #Write  Data to Files
    train.to_csv(data_dir / "processed" / "train.csv", header=True, index=True)
    print(f"\nSuccess! wrote data to {data_dir / "processed" / "train.csv"}")

    test.to_csv(data_dir / "processed" / "test.csv", header=True, index=True)
    print(f"\nSuccess! wrote data to {data_dir / "processed" / "test.csv"}")