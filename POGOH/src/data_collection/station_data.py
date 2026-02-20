import pandas as pd
import numpy as np
import requests
from ckanapi import RemoteCKAN
from io import BytesIO
from pathlib import Path

#--------------------------------------------------------------------------------------
#Helper Functions
#--------------------------------------------------------------------------------------

def get_pogoh_station_data(portal):

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

#--------------------------------------------------------------------------------------
#Main
#--------------------------------------------------------------------------------------
if __name__ == '__main__':

    #Connect to WPRDC
    wprdc = RemoteCKAN('https://data.wprdc.org')

    #Get data
    stations = get_pogoh_station_data(wprdc)
    
    #Get file path
    current_dir = Path(__file__).parent
    root = current_dir.parent.parent
    data_dir = root / "data" / "raw"

    #Write Station Meta-Data to File
    file_path = data_dir / "pogoh_station_data.csv"
    stations.to_csv(file_path, header=True, index=False)
    print(f"\nSuccess! wrote data to {file_path}")

    #Get driving distance data
    distance, duration = get_driving_dist(stations)

    
    #Write Distance Data to Files
    file_path = data_dir / "pogoh_station_dist_miles.csv"
    distance.to_csv(file_path, header=True, index=True)
    print(f"\nSuccess! wrote data to {file_path}")

    file_path = data_dir / "pogoh_station_dist_min.csv"
    duration.to_csv(file_path, header=True, index=True)
    print(f"\nSuccess! wrote data to {file_path}")