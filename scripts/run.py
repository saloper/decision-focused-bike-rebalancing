from dfbr.utils.files import get_config
from dfbr.data.dataset import BikeDemandDataset
from dfbr.models.mlp import MLP
from dfbr.eval.simmulation import Sim, create_station_dict, create_event_df
from dfbr.models.bike_rebalance import BikeRebalanceModel
from dfbr.training.train import get_loss_func, train_one_epoch, evaluate
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.optim import Adam

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Setup
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Read config
config = get_config("baseline.yaml")

#Create a dictionary of stations
station_dict = create_station_dict(config["paths"]["stations"], config["paths"]["station_dist_miles"], config["sim"]["start_inv_pct"])
#Sort by id to ensure alignment
station_ids = sorted(station_dict.keys())

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Load Data for Predictive Model
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Create datasets
train_ds = BikeDemandDataset(
        file = config["paths"]["input"],
        start_date = config["data"]["train_start_date"],
        end_date = config["data"]["train_end_date"],
        target_cols= [str(id) for id in station_ids],
        input_scale_cols= ['mean_temp', 'precip', 'max_gust'],
        input_no_scale_cols=['sin_day_of_week', 'cos_day_of_week', 'sin_month', 'cos_month']
    )

training_stats = {'mean': train_ds.mean, 'std': train_ds.std, 'y_mean': train_ds.y_mean, 'y_std': train_ds.y_std}

test_ds = BikeDemandDataset(
        file = config["paths"]["input"],
        start_date = config["data"]["test_start_date"],
        end_date = config["data"]["test_end_date"],
        target_cols= [str(id) for id in station_ids],
        input_scale_cols= ['mean_temp', 'precip', 'max_gust'],
        input_no_scale_cols=['sin_day_of_week', 'cos_day_of_week', 'sin_month', 'cos_month'],
        is_train=False,
        scaling_factor=training_stats
    )

#Wrap Data Loaders
train_dl = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=False)
test_dl = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Instantiate Model and Train
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Create MLP
input_size = len(train_ds[0][0])
output_size = len(train_ds[0][1])
pred_model = MLP(input_size, output_size, config["model"]["hidden_layers"])

#Create loss function and optimizer
criterion = get_loss_func(config["training"]["loss_function"])
optimizer = Adam(pred_model.parameters(), lr=config["training"]["learning_rate"], weight_decay = config["training"]["weight_decay"])

#Training loop
for epoch in range(config["training"]["epochs"]):
    train_loss = train_one_epoch(pred_model, train_dl, optimizer, criterion, 'cpu')
    test_loss = evaluate(pred_model, test_dl, criterion, 'cpu')
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Run Simulation
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#optimization model
opt_model = BikeRebalanceModel(
    station_file_path=config["paths"]["stations"],
    dist_matrix_file_path=config["paths"]["station_dist_miles"],
    loss_demand_cost=config["model"]["loss_demand_cost"],   
    over_capacity_cost=config["model"]["over_capacity_cost"],  
    movement_cost=config["model"]["movement_cost"]      
)

#Simulation
sim = Sim(
    station_dict= station_dict,
    event_df = create_event_df(config["paths"]["raw_trips"], config["paths"]["stations"], config["data"]["test_start_date"],  config["data"]["test_end_date"], config["sim"]["cutoff_hour"]),
    reset_inv = config["sim"]["resest_inv"],
    reset_inv_pct = config["sim"]["start_inv_pct"],
    predict = config["sim"]["predict"],
    predict_ds = test_ds,
    predict_model = pred_model,
    opt_model = opt_model
)
sim.run()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Collect Metrics and Plot
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
df_metrics = pd.DataFrame({
    'Lost Demand': sim.lost_demand,
    'Over Capacity': sim.over_capacity,
    'Forced Returns': sim.forced_returns,
    'Moved Bikes' : sim.moved_bikes,
    'Total Inventory' : sim.total_inventory,
    })

print(f'System Metrics: \n{df_metrics.mean()}')

axes = df_metrics.plot(
    kind='line',
    subplots=True, 
    figsize=(12, 8), 
    marker='o', 
    alpha=0.8,
    sharex=True
    )

plt.suptitle("System-Wide Daily Metrics", fontsize=14)

# Loop through each of the 3 axes to add gridlines and Y-labels
for ax in axes:
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylabel("Events")

# The X-label only needs to go on the bottom-most plot
plt.xlabel("Date")

plt.tight_layout()
plt.show()

#Plot heatmap of station data
#Extract the nested dictionaries from the station objects
lost_demand_data = {}
over_cap_data = {}
forced_ret_data = {}

for s_id, station in sim.stations.items():
    lost_demand_data[s_id] = station.lost_demand
    over_cap_data[s_id] = station.over_capacity

# 2. Convert to DataFrames and transpose (.T) so Stations are rows and Dates are columns
df_ld = pd.DataFrame(lost_demand_data).T
df_oc = pd.DataFrame(over_cap_data).T

# Ensure the stations (Y-axis) are sorted numerically/alphabetically
df_ld.sort_index(inplace=True)
df_oc.sort_index(inplace=True)

#Set up the matplotlib figure with 3 side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

#Plot Lost Demand (Red)
sns.heatmap(df_ld, ax=axes[0], cmap='Reds', cbar_kws={'label': 'Missed Rentals'})
axes[0].set_title('Lost Demand', fontsize=14)
axes[0].set_ylabel('Station ID', fontsize=12)
axes[0].set_xlabel('Date', fontsize=12)

#Plot Over Capacity (Blue)
sns.heatmap(df_oc, ax=axes[1], cmap='Blues', cbar_kws={'label': 'Failed Returns'})
axes[1].set_title('Over Capacity', fontsize=14)
axes[1].set_ylabel('Station ID', fontsize=12)
axes[1].set_xlabel('Date', fontsize=12)


#Clean up layout and display
plt.tight_layout()
plt.show()

#Flatten the history from all objects into a single list of dictionaries
stations_to_plot = [13,29]

all_records = [
    {'time': record['time'], 'station_id': s.id, 'inventory': record['inventory']}
    for s_id, s in sim.stations.items() 
    if s_id in stations_to_plot
    for record in s.history
]

# 2. Convert to DataFrame
df_inv = pd.DataFrame(all_records)

# 3. Pivot and forward-fill (ffill) to make it a continuous time-series
# Columns will be station_ids, Index will be time, Values will be inventory
df_inv_pivot = df_inv.pivot_table(
    index='time', 
    columns='station_id', 
    values='inventory', 
    aggfunc='last'
).ffill()
df_inv_pivot = df_inv_pivot.loc['2025-01-01': '2025-01-31']
# 4. Plotting

plt.figure(figsize=(14, 6))

for s_id in df_inv_pivot:
   line, = plt.step(
        df_inv_pivot.index, 
        df_inv_pivot[s_id], 
        where='post', # 'post' ensures the step drops/rises exactly at the event time
        label=f'Station {s_id}'
    )
   current_color = line.get_color()
   plt.axhline(y = station_dict.get(s_id).capacity, color=current_color, linestyle='--', label=f'_Station {s_id} Cap')

plt.title('Station Inventory Levels Over Time', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Available Bikes', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()