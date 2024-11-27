import torch
import os
from transformers import BertModel, AutoTokenizer
from settings import Config

class L2EModel(torch.nn.Module):

    def __init__(self, bert_model):
        super(L2EModel, self).__init__()
        self.bert = bert_model

        self.dense_1 = torch.nn.Linear(768, Config.num_class_labels)
        self.dense_2 = torch.nn.Linear(768, Config.num_class_labels*5)


    def forward(self, **kwargs):
        bert_out = self.bert(**kwargs)["last_hidden_state"]

        text_clf_out = self.dense_1(bert_out[:,0])
        span_pred_out = self.dense_2(bert_out[:, 1:-1])
        span_pred_out = span_pred_out.reshape(*span_pred_out.shape[:2], Config.num_class_labels, 5)

        return text_clf_out, span_pred_out

def load_model(method, load_weights_if_available=True):
    bert_model = BertModel.from_pretrained(Config.model_checkpoint)
    model = L2EModel(bert_model)
    tokenizer = AutoTokenizer.from_pretrained(Config.model_checkpoint)

    if load_weights_if_available:
        try:
            model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f"../output_data/saved_models/{Config.dataset_type}/Situ_2021/{Config.save_name}_{method}_weights.pkl")))
            print("Loaded existing model weights!")
        except:
            print("No model weights found!")

    return model, tokenizer

def save_model(model, method):
    os.makedirs(os.path.join(os.path.dirname(__file__), f"../output_data/saved_models/{Config.dataset_type}/Situ_2021/"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), f"../output_data/saved_models/{Config.dataset_type}/Situ_2021/{Config.save_name}_{method}_weights.pkl"))