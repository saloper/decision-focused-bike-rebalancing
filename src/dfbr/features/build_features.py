import numpy as np
import pandas as pd
import requests
from dfbr.utils.files import get_config, get_path

#--------------------------------------------------------------------------------------
#Helper Functions
#--------------------------------------------------------------------------------------
def calc_net_demand(data, freq='D'):
    temp = data.copy()

    #Calculate Net Demand
    outflow = temp.groupby([
        pd.Grouper(key='Start Date', freq=freq), 
        'Start Station Id', 
    ],observed=False).size().rename('Outflow').to_frame()

    inflow = temp.groupby([
        pd.Grouper(key='End Date', freq=freq), 
        'End Station Id', 
    ], observed=False).size().rename('Inflow').to_frame()

    outflow.index.names = ['Start Date', 'Station Id']
    inflow.index.names = ['Start Date', 'Station Id',]

    flow = outflow.join(inflow, how='outer').fillna(0).reset_index()
    flow['Netflow'] =  flow['Outflow'] - flow['Inflow'] 

    print(f"\nSuccess! Combined flow at {freq} frequency. Total rows: {len(flow)}")

    return flow

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

    print(f"\nSuccess! Added temporal encodings.")
    return df

def add_weather_forecast(df):
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 40.4406,
        "longitude": -79.9959,
        "start_date": df.index.min().strftime('%Y-%m-%d'),
        "end_date": df.index.max().strftime('%Y-%m-%d'),
        "daily": [
            "apparent_temperature_mean",
            "precipitation_sum", 
            "wind_gusts_10m_mean"
            ],
        "previous_day" : 1,
        "timezone": 'America/New_York'
    }

    #Make request and get data
    response = requests.get(url, params=params)
    data = response.json()

    
    weather_df = pd.DataFrame(data['daily'])
    weather_df['time'] = pd.to_datetime(weather_df['time']).dt.tz_localize('America/New_York', ambiguous=True)
    weather_df.set_index('time', inplace=True)
    weather_df.columns = ['mean_temp', 'precip', 'max_gust']

    df = df.join(weather_df, how='inner')
    print(f"\nSuccess! Added day-ahead weather forecasts.")

    return df
#--------------------------------------------------------------------------------------
#Main
#--------------------------------------------------------------------------------------
if __name__ == '__main__':
    config = get_config("baseline.yaml")
    
    #Read Raw Files
    trips = pd.read_parquet(config["paths"]["raw_trips"], engine='pyarrow')
    stations = pd.read_parquet(config["paths"]["stations"], engine='pyarrow')
    
    #Filter trips to normal
    trips = trips[trips['Closed Status'] == 'NORMAL']

    #Inner Join to filter out station id changes
    trips['Start Station Name'] = trips['Start Station Name'].str.lower().str.replace(' and ', ' & ', regex=False)
    stations['Name'] = stations['Name'].str.lower().str.replace(' and ', ' & ', regex=False)
    filtered = pd.merge(trips, stations, left_on=['Start Station Id', 'Start Station Name'], right_on=['Id', 'Name'], how='inner')
    
    #Calculate the net flow
    flow = calc_net_demand(filtered)
    flow = flow[['Start Date', 'Station Id', 'Netflow']]

    #Getlist of all IDs
    station_ids = sorted(stations['Id'].unique())
    
    #Pivot the data
    flow = flow.pivot_table(index='Start Date', columns='Station Id', values='Netflow', aggfunc='sum', fill_value=0)
    
    #Force the columns to match station list, filling missing ones with 0
    flow = flow.reindex(columns=station_ids, fill_value=0)

    #force the rows to include all dates
    flow = flow.reindex(pd.date_range(start=flow.index.min(), end=flow.index.max(), freq='D'), fill_value=0)

    #Add in the encoding of the times
    flow = add_temporal_encodings(flow)
    flow = add_weather_forecast(flow)

    #Write  Data to Files
    flow.columns = flow.columns.astype(str)
    flow.to_parquet(config["paths"]["input"], engine='pyarrow')
    print(f"\nSuccess! wrote data to {config["paths"]["input"]}")

