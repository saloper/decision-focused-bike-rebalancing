from dfbr.utils.files import get_config, get_path, setup_logger
from dfbr.data.dataset import BikeDemandDataset, BikeOptTargetsDataset
from dfbr.models.station_targets import BikeStationTargets
from dfbr.models.cost_head import CostHead
from dfbr.models.mlp import MLP
from dfbr.eval.simulation import Sim, create_station_dict, create_event_df
from dfbr.training.train import evaluate
import datetime
import argparse
import shutil
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch
import pyepo

def main():
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Parse command line
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline_healthy_ride.yaml") 
    parser.add_argument("--sim", action="store_true", default=False) 
    args = parser.parse_args()

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Setup
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Read config
    config = get_config(args.config)

    #Get timestamp and create directory to hold data for run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = get_path(f"experiments\\healthy_ride\\{config["experiment_name"]}\\{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    #Make a copy of the configuration used
    shutil.copy(get_path(f"configs/{args.config}"), run_dir / "config.yaml")

    #Setup Logging
    logger = setup_logger(run_dir / "train.log")
    logger.info(f"Starting Experiment: {config["experiment_name"]}")
    logger.info(f"Run directory created at: {run_dir}")


    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Load Datasets
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #Create a dictionary of stations
    station_dict = create_station_dict(config["paths"]["stations"], config["paths"]["station_dist_miles"], config["sim"]["start_inv_pct"])
    #Sort by id to ensure alignment
    station_ids = sorted(station_dict.keys())
    #Get parameters for shapes of datasets and models
    num_stations = len(station_dict)
    capacities = [station_dict[sid].capacity for sid in station_ids]
    max_cap = max(capacities)

    #Create datasets
    train_ds = BikeDemandDataset(
            file = config["paths"]["input"],
            start_date = config["data"]["train_start_date"],
            end_date = config["data"]["train_end_date"],
            target_cols= [str(id) for id in station_ids],
            input_scale_cols= ['mean_temp', 'precip', 'max_gust'],
            input_no_scale_cols=['sin_day_of_week', 'cos_day_of_week', 'sin_month', 'cos_month'],
            capacities=capacities,
            max_cap=max_cap
        )

    training_stats = {'mean': train_ds.mean, 'std': train_ds.std, 'y_mean': train_ds.y_mean, 'y_std': train_ds.y_std}

    test_ds = BikeDemandDataset(
            file = config["paths"]["input"],
            start_date = config["data"]["test_start_date"],
            end_date = config["data"]["test_end_date"],
            target_cols= [str(id) for id in station_ids],
            input_scale_cols= ['mean_temp', 'precip', 'max_gust'],
            input_no_scale_cols=['sin_day_of_week', 'cos_day_of_week', 'sin_month', 'cos_month'],
            capacities=capacities,
            max_cap=max_cap,
            is_train=False,
            scaling_factor=training_stats
        )

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Solve for optimal values with ground truth data
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    opt_model = BikeStationTargets(num_stations=num_stations, max_cap=max_cap, total_inventory=int(sum(capacities) * config["sim"]["start_inv_pct"]))
    pyepo_train_ds = BikeOptTargetsDataset(opt_model, train_ds.X.numpy(), train_ds.c.view(-1, num_stations * (max_cap + 1)).numpy(), train_ds.y.numpy(), train_ds.dates)
    pyepo_test_ds = BikeOptTargetsDataset(opt_model, test_ds.X.numpy(), test_ds.c.view(-1, num_stations * (max_cap + 1)).numpy(), test_ds.y.numpy(), test_ds.dates)

    #Wrap Data Loaders
    train_dl = DataLoader(pyepo_train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    test_dl = DataLoader(pyepo_test_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Instantiate Models and loss functions
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #Create MLP
    input_size = len(train_ds[0][0])
    output_size = num_stations
    pred_model = MLP(input_size, output_size, config["model"]["hidden_layers"])

    #Create cost head
    cost_head = CostHead(capacities=capacities, max_cap=max_cap)
    full_model = nn.Sequential(pred_model, cost_head)

    #Create optimizer
    optimizer = Adam(full_model.parameters(), lr=config["training"]["learning_rate"])
    spo = pyepo.func.SPOPlus(opt_model, processes=4)
    mse = nn.MSELoss()

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Training Loop
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # #Warmstart 
    # #Training loop
    # for epoch in range(config["training"]["epochs"]):
    #     pred_model.train()
    #     #Training Loop
    #     for x, c, w, z, y in train_dl:
    #         #Forward pass
    #         yp = pred_model(x)
    #         loss = mse(yp, y)
    #         #Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
        

    #Training loop
    for epoch in range(config["training"]["epochs"]):
        epoch_train_loss = []
        epoch_test_loss = []

        pred_model.train()
        cost_head.train()
        #Training Loop
        for x, c, w, z, y, _ in train_dl:
            #Forward pass
            yp_scaled = pred_model(x)

            if config["training"]["decision_loss"]:
                #Rescale costs 
                yp_unscaled = (yp_scaled * training_stats["y_std"]) + training_stats["y_mean"]
                cp = cost_head(yp_unscaled)
                loss = spo(cp, c, w, z)
            else:
                loss = mse(yp_scaled, y)
                
            epoch_train_loss.append(loss.item())
            #Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred_model.eval()
        cost_head.eval()
        #Test Loop
        with torch.no_grad():
            for x, c, w, z, y, _ in test_dl:
                #Forward pass
                yp_scaled = pred_model(x)
                if config["training"]["decision_loss"]:
                    #Rescale costs 
                    yp_unscaled = (yp_scaled * training_stats["y_std"]) + training_stats["y_mean"]
                    cp = cost_head(yp_unscaled)
                    loss = spo(cp, c, w, z)
                else:
                    loss = mse(yp_scaled, y)

                epoch_test_loss.append(loss.item())

        logger.info(f"Epoch {epoch+1} Train Loss: {(sum(epoch_train_loss) / len(epoch_train_loss)):.4f} Test Loss: {(sum(epoch_test_loss) / len(epoch_test_loss)):.4f}")
    logger.info("Done training")

    logger.info("Starting evaluation")   
    train_mse, train_cost, opt_train_cost, train_df = evaluate(pred_model, cost_head, opt_model, train_dl, 'train', training_stats)
    test_mse, test_cost, opt_test_cost, test_df = evaluate(pred_model, cost_head, opt_model, test_dl, 'test', training_stats)
    logger.info(f"Final Stats:\nTrain MSE: {train_mse:.4f} Train Cost: {train_cost:.4f}, Optimal Train Cost: {opt_train_cost:.4f}\nTest MSE: {test_mse:.4f} Test Cost: {test_cost:.4f}, Optimal Test Cost: {opt_test_cost:.4f}")

    #Save data
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.sort_values('date').reset_index(drop = True)
    df.to_parquet(run_dir / "results.parquet")
    logger.info(f"Data saved to: {run_dir / 'results.parquet'}")



    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Run Simulation
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if args.sim:
        #Simulation
        sim = Sim(
            station_dict= station_dict,
            station_ids=station_ids,
            event_df = create_event_df(config["paths"]["raw_trips"], config["paths"]["stations"], config["data"]["test_start_date"],  config["data"]["test_end_date"], config["sim"]["cutoff_hour"]),
            num_stations=num_stations,
            max_cap=max_cap,
            predict_ds = test_ds,
            predict_model = pred_model,
            cost_head=cost_head,
            opt_model = opt_model,
            scaling=training_stats
        )
        logger.info("Starting simulation!")
        sim.run()
        logger.info("Starting complete!")

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

        logger.info(f'Simulation Metrics: \n{df_metrics.mean()}')


if __name__ == '__main__':
    main()


# axes = df_metrics.plot(
#     kind='line',
#     subplots=True, 
#     figsize=(12, 8), 
#     marker='o', 
#     alpha=0.8,
#     sharex=True
#     )

# plt.suptitle("System-Wide Daily Metrics", fontsize=14)

# # Loop through each of the 3 axes to add gridlines and Y-labels
# for ax in axes:
#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.set_ylabel("Events")

# # The X-label only needs to go on the bottom-most plot
# plt.xlabel("Date")

# plt.tight_layout()
# plt.show()

# #Plot heatmap of station data
# #Extract the nested dictionaries from the station objects
# lost_demand_data = {}
# over_cap_data = {}
# forced_ret_data = {}

# for s_id, station in sim.stations.items():
#     lost_demand_data[s_id] = station.lost_demand
#     over_cap_data[s_id] = station.over_capacity

# # 2. Convert to DataFrames and transpose (.T) so Stations are rows and Dates are columns
# df_ld = pd.DataFrame(lost_demand_data).T
# df_oc = pd.DataFrame(over_cap_data).T

# # Ensure the stations (Y-axis) are sorted numerically/alphabetically
# df_ld.sort_index(inplace=True)
# df_oc.sort_index(inplace=True)

# #Set up the matplotlib figure with 3 side-by-side subplots
# fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# #Plot Lost Demand (Red)
# sns.heatmap(df_ld, ax=axes[0], cmap='Reds', cbar_kws={'label': 'Missed Rentals'})
# axes[0].set_title('Lost Demand', fontsize=14)
# axes[0].set_ylabel('Station ID', fontsize=12)
# axes[0].set_xlabel('Date', fontsize=12)

# #Plot Over Capacity (Blue)
# sns.heatmap(df_oc, ax=axes[1], cmap='Blues', cbar_kws={'label': 'Failed Returns'})
# axes[1].set_title('Over Capacity', fontsize=14)
# axes[1].set_ylabel('Station ID', fontsize=12)
# axes[1].set_xlabel('Date', fontsize=12)


# #Clean up layout and display
# plt.tight_layout()
# plt.show()

# #Flatten the history from all objects into a single list of dictionaries
# stations_to_plot = [13,29]

# all_records = [
#     {'time': record['time'], 'station_id': s.id, 'inventory': record['inventory']}
#     for s_id, s in sim.stations.items() 
#     if s_id in stations_to_plot
#     for record in s.history
# ]

# # 2. Convert to DataFrame
# df_inv = pd.DataFrame(all_records)

# # 3. Pivot and forward-fill (ffill) to make it a continuous time-series
# # Columns will be station_ids, Index will be time, Values will be inventory
# df_inv_pivot = df_inv.pivot_table(
#     index='time', 
#     columns='station_id', 
#     values='inventory', 
#     aggfunc='last'
# ).ffill()
# df_inv_pivot = df_inv_pivot.loc['2025-01-01': '2025-01-31']
# # 4. Plotting

# plt.figure(figsize=(14, 6))

# for s_id in df_inv_pivot:
#    line, = plt.step(
#         df_inv_pivot.index, 
#         df_inv_pivot[s_id], 
#         where='post', # 'post' ensures the step drops/rises exactly at the event time
#         label=f'Station {s_id}'
#     )
#    current_color = line.get_color()
#    plt.axhline(y = station_dict.get(s_id).capacity, color=current_color, linestyle='--', label=f'_Station {s_id} Cap')

# plt.title('Station Inventory Levels Over Time', fontsize=16)
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Available Bikes', fontsize=12)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()