import yaml


# Function to load the YAML configuration
def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def to_ascii(text: str) -> str:
    return text.encode("ascii", errors="ignore").decode("ascii")
