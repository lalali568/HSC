import yaml

def load_config(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config