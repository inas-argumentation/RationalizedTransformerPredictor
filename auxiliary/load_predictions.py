import os.path
import numpy as np
from settings import Config

def load_predictions(method):
    dir = os.path.join(os.path.dirname(__file__), f"../output_data/predictions/{Config.dataset_type}/{Config.save_name}/{method}")
    predictions = {}
    for filename in os.listdir(dir):
        array = np.load(os.path.join(dir, filename))
        idx, label = filename[:-4].split("_")
        idx, label = int(idx), int(label)
        if idx in predictions:
            predictions[idx][label] = array
        else:
            predictions[idx] = {label: array}
    return predictions
