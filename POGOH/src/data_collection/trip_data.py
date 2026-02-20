import pandas as pd
import requests
import sys
from ckanapi import RemoteCKAN
from io import BytesIO
from pathlib import Path

#--------------------------------------------------------------------------------------
#Helper Functions
#--------------------------------------------------------------------------------------

def get_pogoh_trip_data(portal, months_back=6):

    #Get metadata for the POGOH trip dataset
    package = portal.action.package_show(id='pogoh-trip-data')
    resources = package['resources']
    
    #Sort resources by created date
    resources = sorted(resources, key=lambda x: x.get('created', ''), reverse=True)
    
    all_data = []
    count = 0
    
    for res in resources:
        if count >= months_back:
            break
            
        print(f"Downloading: {res['name']}")
        
        #Download the file into memory
        response = requests.get(res['url'])
        if response.status_code == 200:
            # POGOH typically uses Excel (.xlsx)
            try:
                # Use BytesIO to read the content without saving to disk
                df = pd.read_excel(BytesIO(response.content))
                all_data.append(df)
                count += 1
            except Exception as e:
                print(f" Could not read {res['name']}: {e}")
        else:
            print(f" Failed to download {res['name']}")

    #Combine all months into one big DataFrame
    if len(all_data) > 0:
        final_df = pd.concat(all_data, ignore_index=True)
        print(f"\nSuccess! Fetched {count} months. Total rows: {len(final_df)}")
        return final_df
    else:
        print("No data found.")
        return None

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

    #Connect to WPRDC
    wprdc = RemoteCKAN('https://data.wprdc.org')
    #Get data
    if len(sys.argv) > 1:
        df = get_pogoh_trip_data(wprdc, int(sys.argv[1]))
    else:
        df = get_pogoh_trip_data(wprdc)
    #Calculate net demand
    flow = calc_net_demand(df)
    #Write data to file
    current_dir = Path(__file__).parent
    root = current_dir.parent.parent
    data_dir = root / "data" / "raw"
    file_path = data_dir / "pogoh_trip_data.csv"
    flow.to_csv(file_path, header=True, index=False)
    print(f"\nSuccess! wrote data to {file_path}")