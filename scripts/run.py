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
    event_df= create_event_df(config["paths"]["raw_trips"], config["paths"]["stations"], config["data"]["test_start_date"],  config["data"]["test_end_date"]),
    predict=config["sim"]["predict"],
    predict_ds=test_ds,
    predict_model= pred_model,
    opt_model=opt_model
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