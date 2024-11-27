import numpy as np
from auxiliary.visualize_text import visualize_word_importance

def visualize_processed_annotation(annotation):
    visualize_word_importance(list([(1 if len(x["annotation"]) > 0 else 0, x["word"]) for x in annotation]))

def match_tokens_and_annotation(sample, label, evidences, tokenizer):
    split_text = sample["prediction_text"].split(" ")

    words = []
    max_idx = 0
    for a_idx, (s_idx, e_idx, text) in enumerate(sorted(evidences, key=(lambda x: x[0]))):
        if s_idx < max_idx:
            raise Exception("Overlapping annotations!")
        for token in tokenizer.tokenize(" ".join(split_text[max_idx:s_idx])):
            words.append({"word": token, "annotation": [], "n_tokens": 1})
        for token in tokenizer.tokenize(" ".join(split_text[s_idx:e_idx])):
            words.append({"word": token, "annotation": [a_idx], "n_tokens": 1})
        max_idx = e_idx
    for token in tokenizer.tokenize(" ".join(split_text[max_idx:])):
        words.append({"word": token, "annotation": [], "n_tokens": 1})
    w_idx = 0
    while w_idx < len(words):
        if words[w_idx]["word"][:2] == "##":
            words[w_idx - 1]["word"] += words[w_idx]["word"][2:]
            words[w_idx - 1]["n_tokens"] += 1
            del words[w_idx]
        else:
            w_idx += 1
    return words

def create_gt_annotation(sample, tokenizer):
    label = list(sample["labels"])[0]
    evidences = [(x["start_token"], x["end_token"], x["text"]) for y in sample["evidences"] for x in y]
    annotations = match_tokens_and_annotation(sample, label, evidences, tokenizer)
    return {label: annotations}

def create_gt_annotations(tokenizer, dataset):
    new_annotations = {}
    for idx in dataset.indices["train"] + dataset.indices["val"] + dataset.indices["test"]:
        if not dataset.samples[int(idx)]["index"] == int(idx):
            raise Exception()
        new_annotations[idx] = create_gt_annotation(dataset.get_full_sample(idx), tokenizer)
    return new_annotations