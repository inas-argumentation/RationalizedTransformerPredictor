import torch
import warnings
import sys
import copy
import numpy as np
from tqdm import tqdm
from auxiliary.split_sample import split_sample_and_return_words
from settings import Config

def evaluate_faithfulness(model, tokenizer, dataset, word_scores, split="test"):
    if model is None:
        return -100, -100
    model.eval()
    warnings.filterwarnings("ignore", category=UserWarning)

    # Added randomness is needed as tiebreaker if many words have the same score
    np.random.seed(0)
    mask_randomness = np.random.randn(10000) * 1e-5

    sufficiency_values = []
    comprehensiveness_values = []

    for idx in tqdm(dataset.indices[split], file=sys.stdout, desc="Evaluate faithfulness"):
        sample = dataset.get_full_sample(idx)

        for label in sample["labels"]:
            mask = word_scores[idx][label]
            mask = mask + mask_randomness[:len(mask)]
            sorted_scores = np.sort(mask)[::-1]
            #mask = np.random.permutation(mask)

            words, num_splits, split_counts, n_overlaps = split_sample_and_return_words(tokenizer, sample["prediction_text"])

            label_comp_values = []
            label_suff_values = []
            for test in ["sufficiency", "comprehensiveness"]:
                current_values = []
                for percentage in range(0, 105, 5):
                    words_copy = copy.deepcopy(words)
                    if percentage == 0:
                        score_threshold = sorted_scores[0] + 0.001
                    elif percentage == 100:
                        score_threshold = sorted_scores[-1] - 0.001
                    else:
                        score_threshold = sorted_scores[int(np.round(len(sorted_scores)*(percentage/100)))]

                    # Remove masked words and create new input text
                    for j in range(len(mask)):
                        if (mask[j] <= score_threshold and test == "sufficiency") or (mask[j] > score_threshold and test == "comprehensiveness"):
                            words_copy[j]["tokens"] = ["[PAD]"] * len(words_copy[j]["tokens"])
                    input_texts = [" ".join(["".join(words_copy[n]["tokens"]) for n in range(len(mask)) if s in words_copy[n]["splits"]]) for s in range(num_splits)]

                    # Add pad tokens to make all splits have same length
                    max_length = max(split_counts.values())
                    for j in range(len(input_texts)):
                        input_texts[j] += " [PAD]" * (max_length-split_counts[j])

                    input_texts_tokenized = tokenizer(input_texts, return_tensors='pt', truncation=False).to("cuda")
                    if Config.output_fn == "softmax":
                        prediction = torch.softmax(torch.mean(model(**input_texts_tokenized), dim=0), dim=-1)[label]
                    else:
                        prediction = torch.sigmoid(torch.mean(model(**input_texts_tokenized), dim=0))[label]
                    current_values.append(prediction.detach().cpu().numpy())

                if test == "sufficiency":
                    label_suff_values.append(np.mean([current_values[-1] - c for c in current_values[1:-1]]))
                elif test == "comprehensiveness":
                    label_comp_values.append(np.mean([current_values[0] - c for c in current_values[1:-1]]))
            sufficiency_values.append(np.mean(label_suff_values))
            comprehensiveness_values.append(np.mean(label_comp_values))
    sufficiency = np.mean(sufficiency_values)
    comprehensiveness = np.mean(comprehensiveness_values)
    print("\nFaithfulness results:")
    print(f"Sufficiency:                         {sufficiency:.3f}")
    print(f"Comprehensiveness:                   {comprehensiveness:.3f}")
    return sufficiency, comprehensiveness