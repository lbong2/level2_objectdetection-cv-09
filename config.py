import json

def read_config(path="config.json"):

    with open(path, "r") as f:
        cfg = json.load(f)
        return cfg