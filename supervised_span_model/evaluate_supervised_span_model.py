import torch
from supervised_span_model.model import load_model
from data.load_dataset import load_dataset, load_annotations_and_ground_truth
from supervised_span_model.train_supervised_span_model import evaluate_span_prediction_model

class ClfModel(torch.nn.Module):

    def __init__(self, model):
        super(ClfModel, self).__init__()
        self.model = model

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        return model_output[0]

def evaluate_supervised_span_model(split="test"):
    model, tokenizer = load_model(True)
    model = model.to("cuda")

    dataset = load_dataset()
    annotations, ground_truth_annotation_arrays = load_annotations_and_ground_truth(tokenizer, dataset)

    clf_model = ClfModel(model).to("cuda")

    evaluate_span_prediction_model(model, tokenizer, dataset, split, annotations, ground_truth_annotation_arrays, base_classifier=clf_model)
