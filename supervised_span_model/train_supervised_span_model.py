import random
import warnings
import sys
import numpy as np
import torch
from tqdm import tqdm
from settings import Config
from auxiliary.split_sample import split_sample, split_sample_and_return_words
from auxiliary.loss_fn import categorical_cross_entropy_with_logits, binary_cross_entropy_with_logits
from evaluation.evaluate_classification import evaluate_predictions
from auxiliary.visualize_text import visualize_word_importance
from evaluation.evaluate_span_predictions import evaluate_span_predictions
from supervised_span_model.model import load_model, save_model
from data.load_dataset import load_dataset, load_annotations_and_ground_truth
from evaluation.evaluate_faithfulness import evaluate_faithfulness

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

def encode_sample(text, tokenizer, array=None):
    split_text, split_array, n_overlaps = split_sample(tokenizer, text, max_number_of_tokens_per_split=510, array=array)
    num_splits = len(split_text)

    encoded_dict = tokenizer.batch_encode_plus(
        split_text,
        padding=True,
        return_tensors='pt')
    encoded_dict = {key: tensor.to("cuda") for key, tensor in encoded_dict.items()}
    n_pads = torch.sum(encoded_dict["input_ids"] == 0, dim=-1).tolist()
    if array is not None:
        return encoded_dict, num_splits, split_array, n_overlaps, n_pads
    return encoded_dict, num_splits, n_overlaps, n_pads

def encode_samples(texts, tokenizer, clf_max_batch_size=None, arrays=None):
    split_texts = [split_sample(tokenizer, texts[idx], max_number_of_tokens_per_split=510, array=arrays[idx] if arrays is not None else None) for idx in range(len(texts))]
    num_splits = [len(x[0]) for x in split_texts]

    while clf_max_batch_size is not None and sum(num_splits) > clf_max_batch_size:
        split_texts = split_texts[:-1]
        num_splits = num_splits[:-1]

    all_inputs = [x for a in split_texts for x in a[0]]
    encoded_dict = tokenizer.batch_encode_plus(
        all_inputs,
        padding=True,
        return_tensors='pt')
    encoded_dict = {key: tensor.to("cuda") for key, tensor in encoded_dict.items()}
    if arrays is not None:
        return encoded_dict, num_splits, [e for x in split_texts for e in x[1]], [x[2] for x in split_texts]
    return encoded_dict, num_splits


def evaluate_span_prediction_model(model, tokenizer, dataset, split, annotations, annotation_gt_arrays, print_statistics=True, base_classifier=None):
    with torch.no_grad():
        model.eval()
        indices = dataset.indices[split]

        predictions = []
        token_score_predictions = {}
        labels = []
        for idx in tqdm(indices, desc="Evaluating...", position=0, leave=True, file=sys.stdout):
            sample = dataset.get_full_sample(idx)
            gt_array = annotation_gt_arrays[sample["index"]] if sample["index"] in annotation_gt_arrays else None
            if gt_array is not None:
                input_batch, n_splits, gt_array, overlaps, n_pads = encode_sample(sample["prediction_text"], tokenizer, gt_array)
            else:
                input_batch, n_splits, overlaps, n_pads = encode_sample(sample["prediction_text"], tokenizer, None)
            words = split_sample_and_return_words(tokenizer, sample["prediction_text"])[0]

            model_output = model(**input_batch)

            average_logits = model_output[0].mean(0).np()
            if Config.output_fn == "softmax" or np.sum(average_logits > 0) == 0:
                predictions.append(np.where(average_logits == average_logits.max(), 1, 0).reshape(1, -1))
            else:
                predictions.append((average_logits > 0).astype("float").reshape(1, -1))
            labels.append(sample["one_hot_label"])

            token_score_prediction = torch.swapaxes(torch.sigmoid(model_output[1]), axis0=-1, axis1=-2)
            token_score_prediction = [token_score_prediction[i, :, :token_score_prediction.shape[2]-n_pads[i]] for i in range(token_score_prediction.shape[0])]

            # Merge span predictions from different parts together by linearly blending the overlapping parts.
            merged_prediction = token_score_prediction[0].np()
            for p_idx in range(1, n_splits):
                linear_blend = np.expand_dims(np.linspace(0.01, 1, overlaps[p_idx-1]), axis=0)
                merged_prediction[:, -overlaps[p_idx-1]:] = (merged_prediction[:, -overlaps[p_idx-1]:] * (1-linear_blend)
                                                       + linear_blend * token_score_prediction[p_idx][:, :overlaps[p_idx-1]].np())
                merged_prediction = np.concatenate([merged_prediction, token_score_prediction[p_idx][:, overlaps[p_idx-1]:].np()], axis=-1)

            # Convert prediction made for individual tokens into one for complete words.
            merged_word_prediction = []
            current_idx = 0
            for w in words:
                merged_word_prediction.append(merged_prediction[:, current_idx:current_idx+w["n_tokens"]].mean(-1))
                current_idx += w["n_tokens"]
            merged_word_prediction = np.stack(merged_word_prediction, axis=-1)
            token_score_predictions[sample["index"]] = {l: merged_word_prediction[l] for l in range(Config.num_class_labels)}

            # Print first sample to see progress during training
            if idx == indices[0]:
                print("-------------------------------------------")
                for label in range(Config.num_class_labels):
                    print(f"Label {label}, {sample['one_hot_label'][label]}")
                    visualize_word_importance([(x, w["word"]) for x, w in zip(merged_word_prediction[label], words)])
                #visualize_word_importance([(x, w["word"]) for x, w in zip(merged_word_prediction[np.argmax(sample["one_hot_label"])], words)])

        clf_F1 = evaluate_predictions(np.concatenate(predictions, axis=0), np.stack(labels, axis=0), convert_predictions=False, print_statistics=print_statistics)
        span_scores = evaluate_span_predictions(token_score_predictions, dataset, tokenizer, split)
        if base_classifier is not None:
            evaluate_faithfulness(base_classifier, tokenizer, dataset, token_score_predictions, split=split)
        return span_scores

def train_supervised_span_prediction_model(load_if_available=False):
    model, tokenizer = load_model(load_if_available)
    model = model.to("cuda")

    dataset = load_dataset()
    annotations, ground_truth_annotation_arrays = load_annotations_and_ground_truth(tokenizer, dataset)

    label_sums = np.sum(np.stack([x["one_hot_label"] for x in dataset.samples.values() if x["index"] in dataset.indices["train"]], axis=0), axis=0)
    if Config.output_fn == "softmax":
        class_weights = torch.tensor((1 / label_sums) / (1 / label_sums).mean(), device="cuda")
    else:
        class_weights = 1 / torch.sqrt(torch.tensor(label_sums / len(dataset.indices["train"]), device="cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-6)

    if Config.dataset_type == "Movies":
        score_func = lambda x: float(x[0])
        loss_fn = categorical_cross_entropy_with_logits
    else:
        score_func = lambda x: float(np.mean(x[0]))
        loss_fn = binary_cross_entropy_with_logits

    print("Start training...")
    max_score = score_func(evaluate_span_prediction_model(model, tokenizer, dataset, "val", annotations, ground_truth_annotation_arrays))

    batch_size = 8
    dataset.set_split("train")
    len_dataset = len(dataset)
    batches_per_epoch = int(len_dataset / batch_size)
    sampling_order = []
    loss_avg = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for epoch in range(300):
            print(f"\n\nEpoch {epoch}:")
            model.train()
            bar = tqdm(desc="Loss: None", total=batches_per_epoch, position=0, leave=True, file=sys.stdout)

            for idx in range(batches_per_epoch):
                bar.update(1)
                samples = [dataset.get_full_sample(get_sample_idx(sampling_order, dataset)) for _ in range(batch_size)]
                texts = [x["prediction_text"] for x in samples]
                gt_arrays = [ground_truth_annotation_arrays[x["index"]] if x["index"] in ground_truth_annotation_arrays else None for x in samples]
                batch, splits, gt_arrays, _ = encode_samples(texts, tokenizer, clf_max_batch_size=32, arrays=gt_arrays)
                labels = torch.tensor(np.stack([x["one_hot_label"] for s, x in zip(splits, samples) for _ in range(s)], axis=0), device="cuda")
                prediction = model(**batch)

                clf_loss = loss_fn(prediction[0], labels[:prediction[0].shape[0]], weights=class_weights).mean()
                token_loss = torch.zeros(1, device="cuda")
                for i in range(len(gt_arrays)):
                    if gt_arrays[i] is not None:
                        current_prediction = torch.sigmoid(prediction[1][i][:gt_arrays[i].shape[0]]) * 0.9999 + 0.00005
                        gt = torch.tensor(gt_arrays[i][:, :, 1], device="cuda")
                        token_loss += -(gt * torch.log(current_prediction) + (1-gt) * torch.log(1-current_prediction)).mean()
                        #token_loss += -(gt * torch.log(current_prediction)).mean() + torch.square(((1-gt) * current_prediction).sum(0) / torch.sum(1-gt, dim=0)).sum()

                loss = clf_loss + token_loss / max(1, len([x for x in gt_arrays if x is not None]))
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                loss_avg = exp_avg(float(loss), loss_avg)
                bar.desc = f"Loss: {loss_avg:<.3f}"

            bar.close()

            score = score_func(evaluate_span_prediction_model(model, tokenizer, dataset, "val", annotations, ground_truth_annotation_arrays))
            if score > max_score:
                save_model(model)
                max_score = score
                print("New best! Model saved.")

    print(f"\nMax score: {max_score}")