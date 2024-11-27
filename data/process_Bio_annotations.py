import numpy as np
from auxiliary.visualize_text import visualize_word_importance

def visualize_processed_annotation(annotation):
    visualize_word_importance(list([(1 if len(x["annotation"]) > 0 else 0, x["word"]) for x in annotation]))

label_matching = {
    "ER": 0,
    "BR": 1,
    "PH": 2,
    "DN": 3,
    "IS": 4,
    "LS": 5,
    "PP": 6,
    "DS": 7,
    "IM": 8,
    "TE": 9
}

def get_labels(annotations):
    labels = []
    for annotator in annotations:
        try:
            labels += [label_matching[x["type"][:2]] for x in annotations[annotator]["title annotations"].values()]
            labels += [label_matching[x["type"][:2]] for x in annotations[annotator]["abstract annotations"].values()]
        except:
            print()
    return list(set(labels))

def match(annotation, tokenizer, text, label, offset=0):
    words = []
    max_idx = 0
    for a_idx, a in sorted([(x, y) for x, y in annotation.items() if label_matching[y["type"][:2]] == label], key=(lambda x: x[1]["char_span"][0])):
        if a["char_span"][0] < max_idx:
            if a["char_span"][1] <= max_idx:
                continue
            prev_a_idx = words[-1]["annotation"][0]
            for token in tokenizer.tokenize(text[max_idx:a["char_span"][1]]):
                words.append({"word": token, "annotation": [prev_a_idx], "n_tokens": 1})
            max_idx = a["char_span"][1]
            continue
        for token in tokenizer.tokenize(text[max_idx:a["char_span"][0]]):
            words.append({"word": token, "annotation": [], "n_tokens": 1})
        for token in tokenizer.tokenize(text[a["char_span"][0]:a["char_span"][1]]):
            words.append({"word": token, "annotation": [int(a_idx)+offset], "n_tokens": 1})
        max_idx = a["char_span"][1]
        if len(words) > len(tokenizer.tokenize(text[:max_idx])):
            print()
    for token in tokenizer.tokenize(text[max_idx:]):
        words.append({"word": token, "annotation": [], "n_tokens": 1})
    w_idx = 0
    while w_idx < len(words):
        if words[w_idx]["word"][:2] == "##":
            words[w_idx-1]["word"] += words[w_idx]["word"][2:]
            words[w_idx-1]["n_tokens"] += 1
            del words[w_idx]
        else:
            w_idx += 1
    return words

def match_tokens_and_annotation(annotation, tokenizer, sample, label):
    title_annotations = match(annotation["title annotations"], tokenizer, sample["title"], label)
    abstract_annotations = match(annotation["abstract annotations"], tokenizer, sample["abstract"], label, len(annotation["title annotations"]))
    return title_annotations + [{"word": ".", "annotation": [], "n_tokens": 1}] + abstract_annotations

def create_annotation_intersection(annotation_1, annotation_2):
    annotation = []
    idx_mapping = {}
    for w1, w2 in zip(annotation_1, annotation_2):
        if len(w1["annotation"]) > 0 and len(w2["annotation"]) > 0:
            s = str(w1["annotation"][0]) + str(w2["annotation"][0])
            if s not in idx_mapping:
                idx_mapping[s] = len(idx_mapping)
            a_idx = [idx_mapping[s]]
        else: a_idx = []
        annotation.append({"word": w1["word"], "annotation": a_idx, "n_tokens": w1["n_tokens"]})
    return annotation

def create_annotation_union(annotation_1, annotation_2):
    annotation = []
    prev_idx_1, prev_idx_2, current_a_idx = None, None, -1
    for w1, w2 in zip(annotation_1, annotation_2):
        idx_1 = None if len(w1["annotation"]) == 0 else w1["annotation"][0]
        idx_2 = None if len(w2["annotation"]) == 0 else w2["annotation"][0]
        if idx_1 is None and idx_2 is None:
            annotation.append({"word": w1["word"], "annotation": [], "n_tokens": w1["n_tokens"]})
        elif (None not in [prev_idx_1, idx_1] and prev_idx_1 == idx_1) or\
            (None not in [prev_idx_2, idx_2] and prev_idx_2 == idx_2):
            annotation.append({"word": w1["word"], "annotation": [current_a_idx], "n_tokens": w1["n_tokens"]})
        else:
            current_a_idx += 1
            annotation.append({"word": w1["word"], "annotation": [current_a_idx], "n_tokens": w1["n_tokens"]})
        prev_idx_1, prev_idx_2 = idx_1, idx_2
    return annotation

def create_gt_annotation(annotation, tokenizer, sample):
    annotations = {}
    labels = get_labels(annotation)
    for label in labels:
        label_annotations = []
        for annotator in annotation:
            label_annotations.append(match_tokens_and_annotation(annotation[annotator], tokenizer, sample, label))
        num_annotators = len(annotation)
        if len(set([len(x) for x in label_annotations])) > 1:
            raise Exception("Tokenized text has different lengths for different annotators!")
        if num_annotators == 1:
            annotations[label] = label_annotations[0]
        elif num_annotators == 2: # Only for very rare cases, as usually only one or all three annotators annotated a given sample.
            annotations[label] = create_annotation_union(*label_annotations)
        elif num_annotators == 3:
            annotation_intersections = [create_annotation_intersection(label_annotations[x1], label_annotations[x2]) for x1, x2 in [[0, 1], [0, 2], [1, 2]]]
            annotations[label] = create_annotation_union(create_annotation_union(annotation_intersections[0], annotation_intersections[1]), annotation_intersections[2])
    for label in list(annotations.keys()):
        if len([x for x in annotations[label] if len(x["annotation"]) > 0]) == 0:
            del annotations[label]
    return annotations

def create_gt_annotations(annotations, tokenizer, dataset):
    new_annotations = {}
    for idx in annotations:
        if not dataset.samples[int(idx)]["index"] == int(idx):
            raise Exception()
        new_annotations[int(idx)] = create_gt_annotation(annotations[idx], tokenizer, dataset.samples[int(idx)])
    return new_annotations