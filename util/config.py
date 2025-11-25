import yaml

def load_config_yaml(path):
    """
    Loads a YAML configuration file and returns it as a nested dictionary.

    Args:
        path (str): Path to the config.yaml file.

    Returns:
        dict: Nested dictionary representing the YAML configuration.
    """
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config