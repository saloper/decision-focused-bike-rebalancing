#Main Script to run experiments
import yaml
from pathlib import Path
from torch.optim import Adam
from src.models.mlp import MLP
from src.models.train import get_loss_func, train_one_epoch, evaluate
from src.utils.data_utils import prepare_dataloaders
from src.models.bike_rebalance import BikeRebalanceModel

#--------------------------------------------------------------------------------------
#Helper Functions
#--------------------------------------------------------------------------------------

#Read the config file
def load_config(file="config.yaml"):
    with open(file, "r") as file:
        return yaml.safe_load(file)

#Main program
def main(config=None):

    #Get file paths
    current_dir = Path(__file__).parent
    processed_dir = current_dir / "data" / "processed"
    raw_dir = current_dir / "data" / "raw"
    
    # #Create dataloaders
    # input_size, output_size, train_loader, test_loader = prepare_dataloaders(
    #     train_file_path = processed_dir / "train.csv",
    #     test_file_path = processed_dir / "test.csv",
    #     batch_size = config['models']['batch_size']
    # )

    # #Create model
    # model = MLP(input_size, output_size, config['models']['hidden_layers'])

    # #Create loss function and optimizer
    # criterion = get_loss_func(config['models']['loss_type'])
    # optimizer = Adam(model.parameters(), lr=config['models']['learning_rate'])

    # #Training loop
    # for epoch in range(config['models']['epochs']):
    #     train_loss = train_one_epoch(model, train_loader, optimizer, criterion, 'cpu')
    #     test_loss = evaluate(model, test_loader, criterion, 'cpu')
    #     print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    opt_model = BikeRebalanceModel(raw_dir / "pogoh_station_dist_miles.csv", 0.1 , 1)
    opt_model.set_constraints([0] * 60, [10] * 60)
    print(opt_model.solve())

#--------------------------------------------------------------------------------------
#Main
#--------------------------------------------------------------------------------------
if __name__ == "__main__":
    config = load_config()
    #TODOLoad seeds

    main(config)