import random
import warnings
import sys
import numpy as np
import torch
from weakly_supervised_models.models import load_model, save_model
from tqdm import tqdm
from settings import Config
from auxiliary.split_sample import split_sample, split_sample_and_return_words
from auxiliary.visualize_text import visualize_word_importance
from data.load_dataset import load_annotations_and_ground_truth, load_dataset
from evaluation.evaluate_classification import evaluate_predictions
from evaluation.evaluate_span_predictions import evaluate_span_predictions
from evaluation.evaluate_faithfulness import evaluate_faithfulness
from post_hoc_methods.model import load_model_2
from post_hoc_methods.evaluate_rationales import ClfModel
from auxiliary.generate_latex import generate_html

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

def encode_sample(text, tokenizer, array=None):
    # If the input text is too long, split it up (with overlap between splits).
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

# Evaluate the given model on the specified split. Returns clf-f1 and annotation matching scores.
# If "do_faithfulness_eval", it also evaluates faithfulness of the model and prints the results.
def evaluate_model(model, dataset, split, annotations, annotation_gt_arrays, do_faithfulness_eval=False, print_statistics=True):
    with torch.no_grad():
        model.eval()
        indices = dataset.indices[split]

        predictions = []
        token_score_predictions = {}
        labels = []

        for idx in tqdm(indices, desc="Evaluating...", position=0, leave=True, file=sys.stdout):
            sample = dataset.get_full_sample(idx)
            gt_array = annotation_gt_arrays[sample["index"]] if sample["index"] in annotation_gt_arrays else None

            if gt_array is not None: # This is an annotated sample, for which we calculate span scores.
                input_batch, n_splits, gt_array, overlaps, n_pads = encode_sample(sample["prediction_text"], model.tokenizer, gt_array)
            else: # Some samples in the bio dataset are not annotated with rationales. For these, we only evaluate classification performance.
                input_batch, n_splits, overlaps, n_pads = encode_sample(sample["prediction_text"], model.tokenizer, None)
            words = split_sample_and_return_words(model.tokenizer, sample["prediction_text"])[0]

            model_output = model(input_batch)

            # Create classification output.
            average_logits = model_output[0].mean(0).np()
            if Config.output_fn == "softmax" or np.sum(average_logits > 0) == 0:
                predictions.append(np.where(average_logits == average_logits.max(), 1, 0).reshape(1, -1))
            else:
                predictions.append((average_logits > 0).astype("float").reshape(1, -1))
            labels.append(sample["one_hot_label"])

            token_score_prediction = torch.swapaxes(model_output[1], axis0=-1, axis1=-2)
            token_score_prediction = [token_score_prediction[i, :, :token_score_prediction.shape[2]-n_pads[i]] for i in range(token_score_prediction.shape[0])]

            # Long texts are split into multiple parts. Here, we merge span predictions from different parts together by linearly
            # blending the overlapping parts.
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


            if idx == indices[0]: # Print exemplary output for first sample from dataset to allow for visual monitoring during training.
                print("\n" ,"-------------------------------------------")
                for label in range(Config.num_class_labels):
                    print(f"Label {label}, ({'correct' if sample['one_hot_label'][label] > 0 else 'incorrect'})")
                    visualize_word_importance([(x, w["word"]) for x, w in zip(merged_word_prediction[label], words)])

        # Calculate evaluation scores from all predictions.
        clf_F1 = evaluate_predictions(np.concatenate(predictions, axis=0), np.stack(labels, axis=0), convert_predictions=False, print_statistics=print_statistics)
        span_scores = evaluate_span_predictions(token_score_predictions, dataset, model.tokenizer, split)
        if do_faithfulness_eval:
            s, c = evaluate_faithfulness(model.get_clf_model(), model.tokenizer, dataset, token_score_predictions, split=split)
            #print(f"{clf_F1:.3f} & {span_scores[0]:.3f} & {span_scores[2]:.3f} & {span_scores[4]:.3f} & {span_scores[1]:.3f} & {span_scores[3]:.3f} & {s:.3f} & {c:.3f}")
        return span_scores, clf_F1

# Method to train all weakly supervised methods implemented in "weakly_suervised_models/models.py".
def train_weakly_supervised_model(load_if_available=True, **kwargs):
    # Load the model. If "load_if_available", a trained checkpoint can be loaded.
    model = load_model(kwargs, load_if_available)
    model = model.to("cuda")

    # Loading the dataset (texts and sample labels) and token level annotations.
    dataset = load_dataset()
    annotations, ground_truth = load_annotations_and_ground_truth(model.tokenizer, dataset)

    # Calculating class weights to balance classes in unbalanced datasets.
    label_sums = np.sum(np.stack([x["one_hot_label"] for x in dataset.samples.values() if x["index"] in dataset.indices["train"]], axis=0), axis=0)
    if Config.output_fn == "softmax":
        class_weights = torch.tensor((1 / label_sums) / (1 / label_sums).mean(), device="cuda")
    else:
        class_weights = 1 / torch.sqrt(torch.tensor(label_sums / len(dataset.indices["train"]), device="cuda"))
    model.set_class_weights(class_weights)

    optimizer = torch.optim.AdamW(list(model.parameters()), lr=1e-5, weight_decay=kwargs["weight_decay"])

    # The "score_func" converts the multiple eval scores into a single score for determining the best eval checkpoint.
    # For the movie reviews, the train and eval rationale annotations are more sparse than the test ones, so that we only use the auc score
    # as a rough estimate of rationale quality (since especially the span scores are useless).
    # For the bio dataset, we use the mean over all scores.
    if Config.dataset_type == "Movies":
        score_func = lambda x: float(x[0][0]) + (float(x[1]*3) if kwargs["model_type"] == "Brinner_2024" else 0)
    else:
        score_func = lambda x: float(np.mean(x[0])) + float(x[1]*3)

    print("Start training...")
    max_score = score_func(evaluate_model(model, dataset, "test", annotations, ground_truth))

    dataset.set_split("train")
    len_dataset = len(dataset)
    batches_per_epoch = max(int(len_dataset / TRAIN_BATCH_SIZE), 70)
    sampling_order = []

    loss_avg = None
    correct_list = []
    weight_list = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for epoch in range(kwargs["max_num_epochs"]):
            print(f"\n\nEpoch {epoch}:")
            model.train()
            bar = tqdm(desc="Loss: None", total=batches_per_epoch * TRAIN_BATCH_SIZE, position=0, leave=True, file=sys.stdout)

            for _ in range(batches_per_epoch):
                # We backpropagate through each sample from the batch separately and do gradient accumulation, since each training forward pass
                # sometimes requires many individual model forward passes (e.g., for each rationale), which exceeds memory quickly.
                for _ in range(TRAIN_BATCH_SIZE):
                    bar.update(1)
                    sample = dataset.get_full_sample(get_sample_idx(sampling_order, dataset))

                    # Load the text. If it is too long, split it. Only one split will randomly be used for the update.
                    text = sample["prediction_text"]
                    batch, num_splits, _, _ = encode_sample(text, model.tokenizer)
                    class_labels = torch.tensor(sample["one_hot_label"], device="cuda")

                    # Each model has its own function for calculating the loss for a give sample. Please check out "weakly_suervised_models/models.py".
                    output = model.calculate_training_loss(batch, class_labels, epoch)
                    loss = (1/TRAIN_BATCH_SIZE) * output["loss"]

                    loss_avg = exp_avg(float(loss), loss_avg)
                    correct_list += output["prediction_precision"]
                    if output["average_weights"] is not None:
                        weight_list.append(output["average_weights"])
                    loss.backward()

                torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)

                optimizer.step()
                # Our model uses a frozen copy of itself for scoring the rationales. After a gradient update, this frozen copy is updated.
                model.parameters_updated()
                optimizer.zero_grad()

                correct_list = correct_list[-100:]
                weight_list = weight_list[-100:]
                if epoch >= model.n_warmup_epochs:
                    bar.desc = f"Loss: {loss_avg:.3f}  Clf precision: {float(np.mean(correct_list)):.3f}  Avg. weight: {float(np.mean(weight_list)):.2f}"
                else:
                    bar.desc = f"Warmup...  Loss: {loss_avg:.3f}  Clf precision: {float(np.mean(correct_list)):.3f}"

            bar.close()

            score = score_func(evaluate_model(model, dataset, "val", annotations, ground_truth))
            if score > max_score:
                save_model(kwargs["model_type"], model)
                max_score = score
                print("New best! Model saved.")

    print(f"\nMax val span score: {max_score}")

# Load a model and evaluate it on the specified split. The checkpoint name to be loaded can be specified in the settings.
def evaluate_weakly_supervised_model(model_type="Brinner_2024", split="test"):
    model = load_model({"model_type": model_type}, True)
    model = model.to("cuda")

    dataset = load_dataset()
    annotations, ground_truth = load_annotations_and_ground_truth(model.tokenizer, dataset)

    evaluate_model(model, dataset, split, annotations, ground_truth, do_faithfulness_eval=True)