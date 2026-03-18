from dfbr.utils.files import get_config
from dfbr.data.dataset import BikeDemandDataset
from dfbr.models.mlp import MLP
from dfbr.models.bike_rebalance import BikeRebalanceModel
from dfbr.training.train import get_loss_func, train_one_epoch, evaluate
from torch.utils.data import DataLoader
from torch.optim import Adam

#Read config
config = get_config("baseline.yaml")

#Create datasets
train_ds = BikeDemandDataset(
        file = config["paths"]["input"],
        start_date = config["data"]["train_start_date"],
        end_date = config["data"]["train_end_date"],
        target_cols= [str(id) for id in range(1,61)],
        input_scale_cols= ['mean_temp', 'precip', 'max_gust'],
        input_no_scale_cols=['sin_day_of_week', 'cos_day_of_week', 'sin_month', 'cos_month']
    )

training_stats = {'mean': train_ds.mean, 'std': train_ds.std}

test_ds = BikeDemandDataset(
        file = config["paths"]["input"],
        start_date = config["data"]["test_start_date"],
        end_date = config["data"]["test_end_date"],
        target_cols= [str(id) for id in range(1,61)],
        input_scale_cols= ['mean_temp', 'precip', 'max_gust'],
        input_no_scale_cols=['sin_day_of_week', 'cos_day_of_week', 'sin_month', 'cos_month'],
        is_train=False,
        scaling_factor=training_stats
    )

#Wrap Data Loaders
train_dl = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=False)
test_dl = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)

#Create MLP
input_size = len(train_ds[0][0])
output_size = len(train_ds[0][1])
pred_model = MLP(input_size, output_size, config["model"]["hidden_layers"])

#Create loss function and optimizer
criterion = get_loss_func(config["training"]["loss_function"])
optimizer = Adam(pred_model.parameters(), lr=config["training"]["learning_rate"])

#Training loop
for epoch in range(config["training"]["epochs"]):
    train_loss = train_one_epoch(pred_model, train_dl, optimizer, criterion, 'cpu')
    test_loss = evaluate(pred_model, test_dl, criterion, 'cpu')
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
