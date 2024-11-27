import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from settings import Config

def load_model_2(load_weights_if_available=True):
    model = AutoModelForSequenceClassification.from_pretrained(Config.model_checkpoint, num_labels=Config.num_class_labels)
    tokenizer = AutoTokenizer.from_pretrained(Config.model_checkpoint)

    if load_weights_if_available:
        try:
            model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f"../output_data/saved_models/{Config.dataset_type}/Standard_Clf/{Config.save_name}_weights.pkl")))
            print("Loaded existing model weights!")
        except:
            print("No model weights found!")

    return model, tokenizer

def load_model(load_weights_if_available=True):
    model = AutoModelForSequenceClassification.from_pretrained(Config.model_checkpoint, num_labels=Config.num_class_labels)
    tokenizer = AutoTokenizer.from_pretrained(Config.model_checkpoint)

    if load_weights_if_available:
        try:
            model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f"../output_data/saved_models/{Config.dataset_type}/Standard_Clf/{Config.save_name}_weights.pkl")))
            print("Loaded existing model weights!")
        except:
            print("No model weights found!")

    return model, tokenizer

def save_model(model):
    os.makedirs(os.path.join(os.path.dirname(__file__), f"../output_data/saved_models/{Config.dataset_type}/Standard_Clf/"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), f"../output_data/saved_models/{Config.dataset_type}/Standard_Clf/{Config.save_name}_weights.pkl"))