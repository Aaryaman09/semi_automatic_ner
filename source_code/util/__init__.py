import json, os
from typing import Dict
from datetime import datetime, timedelta

def read_json(path:os.path):
    with open(path, 'r') as file:
        config = json.load(file)
    
    return config

def save_json(dictionary:Dict, path:os.path):
    with open(path, 'w') as file:
        json.dump(dictionary, file, indent=4)

def save_config(updated_source_str:str, updated_model_str:str, updated_prediction_str:str):

    json.loads
    """Save the updated configs to local files."""
    save_json(json.loads(updated_source_str.replace("\'", "\"")), os.path.join('configs','source_data.json'))
    save_json(json.loads(updated_model_str.replace("\'", "\"")), os.path.join('configs','model_config.json'))
    save_json(json.loads(updated_prediction_str.replace("\'", "\"")), os.path.join('configs','prediction_config.json'))
    return "Configurations saved successfully!"

def time_right_now():
    # Get current time in UTC without milliseconds
    current_time_utc = datetime.utcnow().replace(microsecond=0)

    # Calculate 1 hour later without milliseconds
    one_hour_later = current_time_utc + timedelta(hours=1)

    return current_time_utc, one_hour_later
