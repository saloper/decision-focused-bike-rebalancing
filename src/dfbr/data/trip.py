import pandas as pd
import requests
from ckanapi import RemoteCKAN
from io import BytesIO
from dfbr.utils.files import get_path

#--------------------------------------------------------------------------------------
#Helper Functions
#--------------------------------------------------------------------------------------

def get_pogoh_trip_data(portal):

    #Get metadata for the POGOH trip dataset
    package = portal.action.package_show(id='pogoh-trip-data')
    resources = package['resources']
    
    #Sort resources by created date
    resources = sorted(resources, key=lambda x: x.get('created', ''), reverse=True)
    
    all_data = []
    count = 0
    
    for res in resources:
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

def download_pogoh_trip_data(trip_file):

    #Connect to WPRDC
    wprdc = RemoteCKAN('https://data.wprdc.org')
    trips = get_pogoh_trip_data(wprdc)

    #Clean up datatypes
    trips['Start Date'] = pd.to_datetime(trips['Start Date'], errors='coerce').dt.tz_localize('America/New_York', ambiguous=True)
    trips['End Date'] = pd.to_datetime(trips['End Date'], errors='coerce').dt.tz_localize('America/New_York', ambiguous=True)

    categorical_cols = ['Start Station Name', 'End Station Name', 'Closed Status', 'Rider Type']
    for col in categorical_cols:
        if col in trips.columns:
            trips[col] = trips[col].astype('category')

    id_cols = ['Start Station Id', 'End Station Id']
    for col in id_cols:
        if col in trips.columns:
            # Convert to string, replacing 'nan' with actual None/NaN
            trips[col] = pd.to_numeric(trips[col], errors='coerce').astype('Int64')
    
    #Drop missing ids
    trips = trips.dropna(subset=id_cols)
    
    #Sort data
    trips.sort_values(by='Start Date', inplace=True)
    trips.to_parquet(get_path(trip_file), index=False, engine='pyarrow')

    print(f'Success! Trip data written to file.')

#--------------------------------------------------------------------------------------
#Main
#--------------------------------------------------------------------------------------
if __name__ == '__main__':
    download_pogoh_trip_data("data/raw/pogoh_trips.parquet")
