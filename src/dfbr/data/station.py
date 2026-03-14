import pandas as pd
import numpy as np
import requests
from ckanapi import RemoteCKAN
from io import BytesIO
from dfbr.utils.files import get_path

#--------------------------------------------------------------------------------------
#Helper Functions
#--------------------------------------------------------------------------------------

def get_latest_pogoh_stations(portal):

    #Get metadata for the POGOH trip dataset
    package = portal.action.package_show(id='station-locations')
    resources = package['resources']
    
    #Sort resources by created date
    resources = sorted(resources, key=lambda x: x.get('created', ''), reverse=True)
    
    #We only read the latest list of stations
    print(f"Downloading: {resources[0]['name']}")
    
    #Download the file into memory
    response = requests.get(resources[0]['url'])
    if response.status_code == 200:
        try:
            df = pd.read_excel(BytesIO(response.content))
        except Exception as e:
            print(f" Could not read {resources[0]['name']}: {e}")
    else:
        print(f" Failed to download {resources[0]['name']}")

    print(f"\nSuccess! Fetched {resources[0]['name']}. Total rows: {len(df)}")
    return df

def get_driving_dist(data):

    #Concat all station coordinated into a string
    coords_list = [f"{lon},{lat}" for lon, lat in zip(data['Longitude'], data['Latitude'])]
    coords_string = ";".join(coords_list)

    #Create the URL for OSRM API
    url = f"http://router.project-osrm.org/table/v1/driving/{coords_string}?annotations=duration,distance"

    #Make Request
    response = requests.get(url)
    response = response.json()

    if response.get("code") == "Ok":
        distance_matrix_meters = np.array(response["distances"])
        duration_matrix_seconds = np.array(response["durations"])
        
        #Convert to Miles and Minutes
        distance_matrix_miles = distance_matrix_meters * 0.000621371
        duration_matrix_minutes = duration_matrix_seconds / 60.0
        
        station_id = data['Id'].tolist()
    
        df_distances = pd.DataFrame(distance_matrix_miles, index=station_id, columns=station_id).round(2)
        df_durations = pd.DataFrame(duration_matrix_minutes, index=station_id, columns=station_id).round(2)
        
        print(f"\nSuccess! Calculated Distance Matrix with OSRM API. Total rows: {len(df_distances)}")
    else:
        print(f"Error from OSRM: {data.get('message')}")
    
    return df_distances, df_durations

def download_pogoh_station_data(station_file, dist_miles_file, dist_min_file):
    #Connect to WPRDC
    wprdc = RemoteCKAN('https://data.wprdc.org')

    #TODO convert data types to be safe

    #Get Station data
    stations = get_latest_pogoh_stations(wprdc)
    stations.to_parquet(get_path(station_file), index=False, engine='pyarrow')

    #Get driving distance data
    distance, duration = get_driving_dist(stations)
    distance.to_parquet(get_path(dist_miles_file), index=True, engine='pyarrow')
    duration.to_parquet(get_path(dist_min_file), index=True, engine='pyarrow')

    print(f'Success! Station data written to files.')

#--------------------------------------------------------------------------------------
#Main
#--------------------------------------------------------------------------------------
if __name__ == '__main__':
    download_pogoh_station_data("data/raw/pogoh_stations.parquet", "data/raw/pogoh_station_dist_miles.parquet", "data/raw/pogoh_station_dist_min.parquet")