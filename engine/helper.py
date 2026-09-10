import yaml

def load_config(config_path="./configs/gpt2_config.py"):
    """YAML files ko read krke dictionary return krna hai"""
    with open(config_path,"r") as file:
        config = yaml.safe_load(file)
    return config