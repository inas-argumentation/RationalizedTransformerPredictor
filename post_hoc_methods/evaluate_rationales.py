from settings import post_hoc_methods
from auxiliary.load_predictions import load_predictions
from evaluation.evaluate_span_predictions import evaluate_span_predictions
from data.load_dataset import load_dataset
from post_hoc_methods.model import load_model
from evaluation.evaluate_faithfulness import evaluate_faithfulness
import torch

class ClfModel(torch.nn.Module):

    def __init__(self, model):
        super(ClfModel, self).__init__()
        self.model = model

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        return model_output["logits"]

def evaluate_post_hoc_rationales(split="test"):
    dataset = load_dataset()
    model, tokenizer = load_model(True)

    for method in post_hoc_methods:
        print(f"\nMethod: {method}")
        rationales = load_predictions(method)
        span_scores = evaluate_span_predictions(rationales, dataset, tokenizer, split)
        s, c = evaluate_faithfulness(ClfModel(model).to("cuda"), tokenizer, dataset, rationales, split)
        #print(f"- & {span_scores[0]:.3f} & {span_scores[2]:.3f} & {span_scores[4]:.3f} & {span_scores[1]:.3f} & {span_scores[3]:.3f} & {s:.3f} & {c:.3f}")



