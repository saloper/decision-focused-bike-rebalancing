#Main Script to run experiments
import yaml
from src.models.mlp import MLP

#--------------------------------------------------------------------------------------
#Helper Functions
#--------------------------------------------------------------------------------------

#Read the config file
def load_config(file="config.yaml"):
    with open(file, "r") as file:
        return yaml.safe_load(file)

#Main program
def main(args=None):
    print('Running Main!')

#--------------------------------------------------------------------------------------
#Main
#--------------------------------------------------------------------------------------
if __name__ == "__main__":
    config = load_config()
    model = MLP(100,60, config['models']['hidden_layers'])
    print(model)
    main()