import random
import warnings
import sys
import numpy as np
import torch
from tqdm import tqdm
from settings import Config
from auxiliary.split_sample import split_sample
from evaluation.evaluate_classification import evaluate_predictions
from post_hoc_methods.model import load_model, save_model
from data.load_dataset import load_dataset
from auxiliary.loss_fn import categorical_cross_entropy_with_logits, binary_cross_entropy_with_logits

TRAIN_BATCH_SIZE = 32

def to_np(self):
    return self.detach().cpu().numpy()
setattr(torch.Tensor, "np", to_np)

def get_sample_idx(sampling_order, dataset):
    if len(sampling_order) == 0:
        sampling_order = [x for x in dataset.indices["train"]]
        random.shuffle(sampling_order)
    return sampling_order.pop()

def exp_avg(new_val, average, alpha=0.02):
    if average is None:
        average = new_val
    else:
        average = (1-alpha) * average + alpha * new_val
    return average

def encode(text, tokenizer):
    split_text, split_array, n_overlaps = split_sample(tokenizer, text, max_number_of_tokens_per_split=510, array=None)
    num_splits = len(split_text)

    encoded_dict = tokenizer.batch_encode_plus(
        split_text,
        padding=True,
        return_tensors='pt')
    encoded_dict = {key: tensor.to("cuda") for key, tensor in encoded_dict.items()}
    return encoded_dict, num_splits

def evaluate_model(model, tokenizer, dataset, split, print_statistics=True):
    model.eval()

    predictions = []
    labels = []
    for idx in tqdm(dataset.indices[split], desc="Evaluating...", position=0, leave=True, file=sys.stdout):
        sample = dataset.get_full_sample(idx)
        input_batch, n_splits = encode(sample["prediction_text"], tokenizer)

        model_output = model(**input_batch)["logits"]
        average_logits = model_output.mean(0).np()
        if Config.output_fn == "softmax":
            predictions.append(np.where(average_logits == average_logits.max(), 1, 0).reshape(1, -1))
        else:
            predictions.append((average_logits > 0).astype("float").reshape(1, -1))
        labels.append(sample["one_hot_label"])

    clf_F1 = evaluate_predictions(np.concatenate(predictions, axis=0), np.stack(labels, axis=0), convert_predictions=False, print_statistics=print_statistics)
    return clf_F1

def train_classifier(load_if_available=False):
    model, tokenizer = load_model(load_if_available)
    model = model.to("cuda")

    dataset = load_dataset()

    label_sums = np.sum(np.stack([x["one_hot_label"] for x in dataset.samples.values() if x["index"] in dataset.indices["train"]], axis=0), axis=0)
    if Config.output_fn == "softmax":
        class_weights = torch.tensor((1 / label_sums) / (1 / label_sums).mean(), device="cuda")
    else:
        class_weights = 1 / torch.sqrt(torch.tensor(label_sums / len(dataset.indices["train"]), device="cuda"))

    optimizer = torch.optim.AdamW(list(model.parameters()), lr=4e-5)
    loss_fn = binary_cross_entropy_with_logits if Config.output_fn == "sigmoid" else categorical_cross_entropy_with_logits

    print("Start training...")
    max_f1 = evaluate_model(model, tokenizer, dataset, "val")

    dataset.set_split("train")
    len_dataset = len(dataset)
    batches_per_epoch = max(int(len_dataset / TRAIN_BATCH_SIZE), 50)
    sampling_order = []

    loss_avg = None
    epochs_without_improvement = 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for epoch in range(80):
            print(f"\n\nEpoch {epoch}:")
            model.train()
            bar = tqdm(desc="Loss: None", total=batches_per_epoch * TRAIN_BATCH_SIZE, position=0, leave=True, file=sys.stdout)

            for _ in range(batches_per_epoch):

                for _ in range(TRAIN_BATCH_SIZE):
                    bar.update(1)
                    sample = dataset.get_full_sample(get_sample_idx(sampling_order, dataset))
                    prediction_text = sample["prediction_text"]
                    batch, num_splits = encode(prediction_text, tokenizer)
                    class_labels = torch.tensor(sample["one_hot_label"], device="cuda")

                    model_output = model(**batch)["logits"]
                    average_logits = torch.mean(model_output, dim=0, keepdim=True)
                    loss = loss_fn(average_logits, class_labels.unsqueeze(0), class_weights)

                    loss_avg = exp_avg(float(loss), loss_avg)
                    loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                bar.desc = f"Loss: {loss_avg:.3f}"

            bar.close()

            f1 = evaluate_model(model, tokenizer, dataset, "val")
            if f1 > max_f1:
                save_model(model)
                max_f1 = f1
                print("New best! Model saved.")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement == 8:
                break

    print(f"\nMax val f1 score: {max_f1}")

def evaluate_base_classifier(split="test"):
    model, tokenizer = load_model(True)
    model = model.to("cuda")

    dataset = load_dataset()
    f1 = evaluate_model(model, tokenizer, dataset, split)