from dfbr.data.station import download_pogoh_station_data
from dfbr.data.trip import download_pogoh_trip_data
from dfbr.utils.files import get_config

#Read config
config = get_config("baseline.yaml")
#Read stations
download_pogoh_station_data(config["paths"]["stations"], config["paths"]["station_dist_miles"], config["paths"]["station_dist_min"] )
#Read trips
download_pogoh_trip_data(config["paths"]["raw_trips"])