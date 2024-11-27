import os
import os.path
import numpy as np
import spacy
from data.load_dataset import load_annotations_and_ground_truth
from settings import Config
from data.process_Bio_annotations import create_gt_annotations
from sklearn.metrics import precision_recall_curve, auc
from auxiliary.visualize_text import visualize_word_importance

def calc_AUC_score(gt, pred):
    precision, recall, thresholds = precision_recall_curve(gt, pred)
    score = auc(recall, precision)
    return score

def calc_F1(precision, recall):
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

spacy_model = None
def get_sentence_split_for_sample(sample):
    os.makedirs(os.path.join(os.path.dirname(__file__), f"temp/{Config.dataset_type}/sentence_split/"), exist_ok=True)
    if not os.path.exists(os.path.join(os.path.dirname(__file__), f"temp/{Config.dataset_type}/sentence_split/{sample['index']}.txt")):
        global spacy_model
        if spacy_model is None:
            spacy_model = spacy.load("en_core_sci_scibert")
        doc = spacy_model(sample["prediction_text"])
        sentences = [str(x) for x in doc.sents]
        with open(os.path.join(os.path.dirname(__file__), f"temp/{Config.dataset_type}/sentence_split/{sample['index']}.txt"), "w+") as f:
            f.write("\t".join(sentences))
    else:
        with open(os.path.join(os.path.dirname(__file__), f"temp/{Config.dataset_type}/sentence_split/{sample['index']}.txt"), "r") as f:
            sentences = f.read().split("\t")
    return sentences

def extract_spans(array, split_points=None):
    spans = []
    current_span = None
    for i in range(len(array)):
        if array[i] == 1:
            split = split_points is not None and i in split_points
            if current_span is None and not split:
                current_span = [i, i+1]
            elif not split:
                current_span[1] += 1
            elif current_span is not None:
                spans.append([current_span[0], current_span[1]+0])     # Omit punctuation? Otherwise +1
                current_span = None
        else:
            if current_span is not None:
                spans.append(current_span)
                current_span = None
    if current_span is not None:
        spans.append(current_span)
    return spans

def calc_IoU_between_spans(spans_1, spans_2):
    IoU = np.zeros((len(spans_1), len(spans_2)), dtype="float32")
    for i in range(len(spans_1)):
        for j in range(len(spans_2)):
            max_min_val = max(spans_1[i][0], spans_2[j][0])
            min_max_val = min(spans_1[i][1], spans_2[j][1])
            n_overlap = float(max(0, min_max_val - max_min_val))
            n_union = float(len(set(list(range(spans_1[i][0], spans_1[i][1])) + list(range(spans_2[j][0], spans_2[j][1])))))
            IoU[i, j] = n_overlap / n_union
    return IoU

def calc_token_F1(pred, gt):
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return calc_F1(precision, recall)

def calc_F1_and_span_F1_score(gt_spans, gt_array, pred, percentage, sentence_split_points, sorted_pred):
    threshold = sorted_pred[int((1-percentage)*pred.shape[0])]
    selection = (pred > threshold).astype('int')

    # Calculate continuous IoU F1 score
    pred_spans = extract_spans(selection, sentence_split_points)
    IoU = calc_IoU_between_spans(gt_spans, pred_spans)
    IoU_precision = np.mean(np.max(IoU, axis=0))
    IoU_recall = np.mean(np.max(IoU, axis=-1))
    IoU_F1 = calc_F1(IoU_precision, IoU_recall)

    # Calculate token F1 score
    token_F1 = calc_token_F1(selection, gt_array)

    return IoU_F1, token_F1

def calc_average_F1_and_span_F1_score(gt, pred, sample, tokenizer):
    pred = pred + np.linspace(0, 1, pred.shape[0]) * 1e-5

    if Config.dataset_type == "Bio":
        # For Bio dataset, set last token of every sentence (punctuation) to zero, as annotated spans never cross sentence boundaries.
        sentences = get_sentence_split_for_sample(sample)
        num_words_per_sentence = [len([x for x in tokenizer.tokenize(s) if x[:2] != "##"]) for s in sentences]
        sentence_split_points = [num_words_per_sentence[0]-1]
        for n in num_words_per_sentence[1:]:
            sentence_split_points.append(sentence_split_points[-1] + n)

        for s in sentence_split_points:
            pred[s] = 0
    sentence_split_points = []
    gt_spans = extract_spans(gt)

    # Calculate discrete span and token F1 scores
    threshold = np.sort(pred)[-int(gt.sum())]
    #threshold = threshold - 0.2*np.sqrt(np.var(pred))
    selection = (pred >= threshold).astype("int")
    pred_spans = extract_spans(selection, sentence_split_points)
    if len(pred_spans) > 0:
        IoU = calc_IoU_between_spans(gt_spans, pred_spans)
        IoU = (IoU > 0.5).astype("int")
        tp = IoU.sum()
        precision = tp / len(pred_spans)
        recall = tp / len(gt_spans)
        discrete_IoU_F1 = calc_F1(precision, recall)
    else:
        discrete_IoU_F1 = 0
    discrete_token_F1 = calc_token_F1(selection, gt)

    # Calculate token F1 score and continuous span F1 score
    IoU_F1_scores = []
    token_F1_scores = []
    sorted_pred = np.sort(pred)
    for percentage in np.linspace(0.05, 0.95, 19):
        IoU_F1, token_F1 = calc_F1_and_span_F1_score(gt_spans, gt, pred, percentage, sentence_split_points, sorted_pred)
        IoU_F1_scores.append(IoU_F1)
        token_F1_scores.append(token_F1)
    return np.mean(IoU_F1_scores), np.mean(token_F1_scores), discrete_IoU_F1, discrete_token_F1

def evaluate_span_predictions(predictions, dataset, tokenizer, split="test"):
    annotations, ground_truth_annotation_arrays = load_annotations_and_ground_truth(tokenizer, dataset)

    indices = dataset.indices[split]
    indices = [x for x in indices if x in annotations]

    scores = []
    for idx in indices:
        sample = dataset.get_full_sample(idx)
        gt = annotations[idx]

        current_sample_scores = []
        for label in gt:
            pred = predictions[idx][label]

            #pred = np.random.random(pred.shape)
            gt_array = np.array([1 if len(x["annotation"]) > 0 else 0 for x in gt[label]])
            current_sample_scores.append([calc_AUC_score(gt_array, pred), *calc_average_F1_and_span_F1_score(gt_array, pred, sample, tokenizer)])

        if False: # Uncomment to visualize predictions and annotations
            words = tokenizer.tokenize(sample["prediction_text"])
            i = 1
            while i < len(words):
                if words[i][:2] == "##":
                    words[i-1] += words[i][2:]
                    del words[i]
                else:
                    i += 1
            visualize_word_importance(list(zip(pred, gt_array, words)))
            print()
        if len(gt) > 0:
            scores.append(np.mean(np.array(current_sample_scores), axis=0))
            #print(sample["index"], scores[-1])

    mean_scores = np.mean(np.array(scores), axis=0)
    print(f"Average auc score:                   {mean_scores[0]:.3f}")
    print(f"Average span IoU F1 score:           {mean_scores[1]:.3f}")
    print(f"Discrete span IoU F1 score:          {mean_scores[3]:.3f}")
    print(f"Average token F1 score:              {mean_scores[2]:.3f}")
    print(f"Discrete token F1 score:             {mean_scores[4]:.3f}")

    #print(f"{mean_scores[0]:.3f} & {mean_scores[2]:.3f} & {mean_scores[1]:.3f} & {mean_scores[4]:.3f} & {mean_scores[3]:.3f}")

    return mean_scores