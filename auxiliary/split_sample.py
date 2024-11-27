import numpy as np

def split_sample(tokenizer, text, max_number_of_tokens_per_split=510, array=None):
    text_tokenized = tokenizer.tokenize(text)
    n_overlap = 100

    n_splits = int(np.ceil(max(len(text_tokenized) - n_overlap, 1) / (max_number_of_tokens_per_split - n_overlap)))
    if n_splits == 1:
        splits = [text_tokenized]
        if array is not None:
            split_arrays = [array]
        n_overlaps = []
    else:
        tokens_per_split = int(np.ceil((len(text_tokenized) + (n_splits-1) * n_overlap) / n_splits))
        n_overlaps = []

        splits = [text_tokenized[:tokens_per_split]]
        if array is not None:
            split_arrays = [array[:tokens_per_split]]
        pos = tokens_per_split - n_overlap
        for i in range(1, n_splits):
            splits.append(text_tokenized[pos:pos+tokens_per_split])
            if array is not None:
                split_arrays.append(array[pos:pos+tokens_per_split])
            pos = pos + tokens_per_split - n_overlap
            n_overlaps.append(n_overlap)

    for i in range(1, len(splits)):
        while splits[i][0][:2] == "##":
            del splits[i][0]
            if array is not None:
                split_arrays[i] = split_arrays[i][1:]
            n_overlaps[i-1] -= 1

    split_texts = [tokenizer.convert_tokens_to_string(x) for x in splits]
    return split_texts, split_arrays if array is not None else [None], n_overlaps

def create_word_list(text_tokenized, splits):
    words = [{"word": text_tokenized[0], "n_tokens": 1, "tokens": [text_tokenized[0]], "splits": [0]}]
    for idx in range(1, len(text_tokenized)):
        token = text_tokenized[idx]
        token_splits = [s for s in range(len(splits)) if idx in splits[s]]
        if token[:2] == "##":
            words[-1]["word"] += token[2:]
            words[-1]["n_tokens"] += 1
            words[-1]["tokens"].append(token[2:])
            words[-1]["splits"] = [x for x in words[-1]["splits"] if x in token_splits]
        else:
            words.append({"word": token,
                          "n_tokens": 1,
                          "tokens": [token],
                          "splits": token_splits})

    return words

def split_sample_and_return_words(tokenizer, text, max_number_of_tokens_per_split=510, n_overlap=100):
    text_tokenized = tokenizer.tokenize(text)

    words = create_word_list(text_tokenized, [list(range(len(text_tokenized)))])

    n_splits = int(np.ceil(max(len(text_tokenized) - n_overlap, 1) / (max_number_of_tokens_per_split - n_overlap)))
    if n_splits == 1:
        splits = [list(range(len(text_tokenized)))]
    else:
        tokens_per_split = int(np.ceil((len(text_tokenized) + (n_splits-1) * n_overlap) / n_splits))

        text_indices = list(range(len(text_tokenized)))
        splits = [text_indices[:tokens_per_split]]
        pos = tokens_per_split - n_overlap
        for i in range(1, n_splits):
            splits.append(text_indices[pos:pos+tokens_per_split])
            pos = pos + tokens_per_split - n_overlap

        token_idx = 0
        for i in range(len(words)):
            words[i]["splits"] = [s for s in range(len(splits)) if token_idx in splits[s] and token_idx + words[i]["n_tokens"] in splits[s]]
    n_overlaps = [n_overlap] * (len(splits)-1)

    #split_text = tokenizer.convert_tokens_to_string(text_tokenized)
    for i in range(1, len(splits)):
        while text_tokenized[splits[i][0]] == "##":
            del splits[i][0]
            n_overlaps[i-1] -= 1

    words = create_word_list(text_tokenized, splits)
    return words, len(splits), {i: sum([x["n_tokens"] for x in words if i in x["splits"]]) for i in range(len(splits))}, n_overlaps