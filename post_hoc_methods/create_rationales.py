import os
import numpy as np
import torch

from settings import post_hoc_methods, Config
from post_hoc_methods.model import load_model
from post_hoc_methods.create_rationale_captum import set_interpretability_approach, create_rationale_captum
from post_hoc_methods.create_rationale_MaRC import create_rationale_MaRC
from data.load_dataset import load_dataset, load_annotations_and_ground_truth
from tqdm import tqdm

def create_rationales_for_all_methods(split="test", methods=None):
    base_clf, tokenizer = load_model(True)
    base_clf = base_clf.to("cuda")
    dataset = load_dataset()
    indices = dataset.indices[split]

    #annotations, ground_truth_annotation_arrays = load_annotations_and_ground_truth(tokenizer, dataset)

    for method in (post_hoc_methods if methods is None else methods):
        print(f"Predicting masks for {method} method.")
        os.makedirs(os.path.join(os.path.dirname(__file__), f"../output_data/predictions/{Config.dataset_type}/{Config.save_name}/{method}"), exist_ok=True)
        set_interpretability_approach(method, base_clf)

        for i, sample_idx in tqdm(list(enumerate(indices)), desc=f"Method: {method}"):
            sample = dataset.get_full_sample(sample_idx)
            for label in sample["labels"]:
                if os.path.exists(os.path.join(os.path.dirname(__file__), f"../output_data/predictions/{Config.dataset_type}/{Config.save_name}/{method}/{sample_idx}_{label}.npy")):
                    continue
                if method == post_hoc_methods[0]:
                    while True:
                        rationale = create_rationale_MaRC(base_clf, tokenizer, (sample["prediction_text"], label))
                        if rationale is not None and np.mean(rationale) < 0.9: break
                else:
                    rationale = create_rationale_captum((sample["prediction_text"], label), tokenizer)
                np.save(os.path.join(os.path.dirname(__file__), f"../output_data/predictions/{Config.dataset_type}/{Config.save_name}/{method}/{sample_idx}_{label}.npy"), rationale)