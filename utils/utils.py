import yaml


# Function to load the YAML configuration
def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)
