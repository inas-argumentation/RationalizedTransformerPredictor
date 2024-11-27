from settings import Config
from evaluation.evaluate_span_predictions import evaluate_span_predictions
from data.load_dataset import load_dataset
from transformers import AutoTokenizer
import os
import numpy as np

def load_predictions(method):
    dir = os.path.join(os.path.dirname(__file__), f"../output_data/predictions/{Config.dataset_type}/{Config.save_name}/{method}")
    predictions = {}
    for filename in os.listdir(dir):
        array = np.load(os.path.join(dir, filename))
        idx = int(filename[:-4])
        predictions[idx] = {label: array[label] for label in range(Config.num_class_labels)}
    return predictions

def evaluate_A2R_rationales(split="test"):
    dataset = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(Config.model_checkpoint)

    for method in ["A2R", "A2R-Noise"]:
        print(f"\nMethod: {method}")
        rationales = load_predictions(method)
        span_scores = evaluate_span_predictions(rationales, dataset, tokenizer, split)
        #print(f"??? & {span_scores[0]:.3f} & {span_scores[2]:.3f} & {span_scores[4]:.3f} & {span_scores[1]:.3f} & {span_scores[3]:.3f} & ??? & ???")



