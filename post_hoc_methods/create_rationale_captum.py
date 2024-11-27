# Testing interpretability approaches that can be implemented by using the captum library
import torch
import numpy as np
import warnings
from settings import post_hoc_methods
from captum.attr import InputXGradient, Saliency, Occlusion, IntegratedGradients, LimeBase, ShapleyValueSampling
from captum._utils.models.linear_model import SkLearnLinearModel
from auxiliary.split_sample import split_sample_and_return_words
from auxiliary.visualize_text import visualize_word_importance

interpretability_approach = None
global_LIME_background = None
global_LIME_label_transform = None
model_attr, attr = None, None

# Needed for LIME implementation.
def to_interp_transform_LIME(current_sample, original_input):
    sample = torch.all(current_sample == original_input, dim=-1).float()
    return sample

# Needed for LIME implementation. Creates a perturbed sample where small spans of words are replaced by [PAD] tokens.
def perturb_func_LIME(original_input):
    mask_percentage = np.random.rand() * 0.08 + 0.05
    random_mask = (np.random.rand(original_input.shape[0], original_input.shape[1]) > mask_percentage).astype("int32")
    for i in range(3):
        random_mask[:, 1:] = random_mask[:, 1:] * random_mask[:, :-1]
    random_mask = torch.tensor(random_mask, device="cuda", dtype=torch.float32).unsqueeze(-1)
    return original_input * random_mask + global_LIME_background * (1 - random_mask)

# Needed for LIME implementation.
def similarity_kernel_LIME(original_input, perturbed_input, perturbed_interpretable_input, **kwargs):
    return torch.mean(perturbed_interpretable_input, dim=-1)

class BertModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model

    def forward(self, input, attention_mask, labels):
        output = self.model(inputs_embeds=input, attention_mask=attention_mask)[0]
        if global_LIME_label_transform is not None:
            output = output[:, global_LIME_label_transform].unsqueeze(0)
        return output

# Initialize model and interpretability function
def set_interpretability_approach(approach, model):
    global interpretability_approach
    interpretability_approach = approach

    if interpretability_approach != post_hoc_methods[0]:
        global model_attr, attr
        model_attr = BertModelWrapper(model)
        if interpretability_approach == post_hoc_methods[1]:
            attr = Occlusion(model_attr)
        elif interpretability_approach in post_hoc_methods[2:4]:
            attr = Saliency(model_attr)
        elif interpretability_approach in post_hoc_methods[4:6]:
            attr = InputXGradient(model_attr)
        elif interpretability_approach == post_hoc_methods[6]:
            interpretable_model = SkLearnLinearModel("linear_model.Ridge")
            attr = LimeBase(model_attr, interpretable_model, similarity_func=similarity_kernel_LIME, perturb_func=perturb_func_LIME,
                        perturb_interpretable_space=False, from_interp_rep_transform=None, to_interp_rep_transform=to_interp_transform_LIME)
        elif interpretability_approach in post_hoc_methods[7:9]:
            attr = IntegratedGradients(model_attr)
        elif interpretability_approach == post_hoc_methods[9]:
            attr = ShapleyValueSampling(model_attr)
        else:
            raise Exception("Interpretability approach must be one of the ones specified in settings.py")

def create_rationale_captum(sample, tokenizer):
    global global_LIME_label_transform
    global_LIME_label_transform = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        model_attr.eval()
        words, num_splits, split_word_counts, n_overlaps = split_sample_and_return_words(tokenizer, sample[0])

        # Lists how many tokens each word has. A token with count -1 indicates a token with no influence on the prediction (e.g., PAD, CLS).
        split_word_tokens_counts = [[-1] for _ in range(num_splits)]
        split_words = [list() for _ in range(num_splits)]
        for word in words:
            for split in word["splits"]:
                split_word_tokens_counts[split].append(word["n_tokens"])
                split_words[split].append(word["word"])

        # Append [PAD] tokens to split_words[split] list to make all splits have the same number of tokens.
        max_split_length = max(split_word_counts.values())
        num_pads_per_split = []
        for i in range(num_splits):
            if (diff := max_split_length - split_word_counts[i]) > 0:
                split_word_tokens_counts[i] = split_word_tokens_counts[i] + [-1] * diff
                split_words[i] = split_words[i] + ["[PAD]"] * diff
            num_pads_per_split.append(diff)

        all_word_tokens_counts = []
        for i in range(num_splits):
            split_word_tokens_counts[i].append(-1)
            all_word_tokens_counts = all_word_tokens_counts + split_word_tokens_counts[i]

        sample_tokenized = tokenizer(split_words, return_tensors='pt', truncation=False, is_split_into_words=True)
        # Use sequence of PAD token embeddings as uninformative input
        background = torch.clone(sample_tokenized["input_ids"])
        #background[torch.logical_not(torch.logical_or(background == 2, background == 3))] = tokenizer.pad_token_id
        control_tokens = torch.isin(background, torch.tensor(tokenizer.all_special_ids))
        background[torch.logical_not(control_tokens)] = tokenizer.pad_token_id

        embeddings_sample = model_attr.model.bert.embeddings(sample_tokenized["input_ids"].to("cuda"))
        embeddings_background = model_attr.model.bert.embeddings(background.to("cuda"))

        label = sample[1]
        print(f"Predicting for label {label}")
        fwd = (sample_tokenized["attention_mask"].to("cuda"), None)
        if interpretability_approach == post_hoc_methods[1]: # Occlusion
            # Window width needs to be uneven number
            attributions = attr.attribute(embeddings_sample, (4 + (1 if embeddings_sample.shape[1] % 2 == 0 else 0), 768), (1, 0), embeddings_background, target=label, additional_forward_args=fwd)
            final_weights = attributions[:, 1:-1, 0]
        elif interpretability_approach in post_hoc_methods[2:4]: # Saliency
            attributions = attr.attribute(embeddings_sample, target=label, additional_forward_args=fwd, abs=False)
        elif interpretability_approach in post_hoc_methods[4:6]:  # InputXGrad
            attributions = attr.attribute(embeddings_sample, target=label, additional_forward_args=fwd)
        elif interpretability_approach == post_hoc_methods[6]:  # LIME
            global global_LIME_background
            attributions = None
            for split in range(num_splits):
                global_LIME_background = embeddings_background[split].unsqueeze(0)
                global_LIME_label_transform = label
                new_attributions = attr.attribute(embeddings_sample[split].unsqueeze(0),
                                              additional_forward_args=(sample_tokenized["attention_mask"][split].unsqueeze(0).to("cuda"), None), n_samples=100)
                if attributions is None:
                    attributions = new_attributions
                else:
                    attributions = torch.concat([attributions, new_attributions], dim=0)
            final_weights = attributions[:, 1:-1]
        elif interpretability_approach in post_hoc_methods[7:9]:  # Integrated Gradients
            attributions = attr.attribute(embeddings_sample, embeddings_background, target=label, additional_forward_args=fwd, internal_batch_size=4)
        elif interpretability_approach == post_hoc_methods[9]:   # Shapley Values
            attributions = None
            for split in range(num_splits):
                background = embeddings_background[split].unsqueeze(0)
                feature_mask = torch.tensor(np.array(list(range(embeddings_sample.shape[1]))).reshape(1, -1, 1), device="cuda", dtype=torch.int)
                new_attributions = attr.attribute(embeddings_sample[split].unsqueeze(0), background,
                                                  feature_mask=feature_mask, target=label, perturbations_per_eval=8,
                                                  additional_forward_args=(sample_tokenized["attention_mask"][split].unsqueeze(0).to("cuda"), None), n_samples=15)
                if attributions is None:
                    attributions = new_attributions
                else:
                    attributions = torch.concat([attributions, new_attributions], dim=0)
            final_weights = attributions[:, 1:-1, 0]

        # If the interpretability approach returned a score for each input embedding element, we reduce it to a single value per token.
        if "L1" in interpretability_approach:
            final_weights = attributions.norm(p=1, dim=-1)[:, 1:-1]
        elif "L2" in interpretability_approach:
            final_weights = attributions.norm(p=2, dim=-1)[:, 1:-1]
        elif "Sum" in interpretability_approach:
            final_weights = attributions.sum(-1)[:, 1:-1]

        # If a word consists of multiple tokens, take the mean of the token scores as score for the whole word.
        final_weights = final_weights.detach().cpu().numpy()
        final_weights_by_words = []
        for i in range(num_splits):
            f = final_weights[i]
            r = []
            c = 0
            n = len([x for x in split_word_tokens_counts[i] if x != -1])
            while len(r) < n:
                r.append(np.mean(f[c:c+split_word_tokens_counts[i][1+len(r)]]))
                c += split_word_tokens_counts[i][len(r)]
            final_weights_by_words.append(np.array(r))

        # If the sample was split into multiple parts, merge individual mask parts by blending overlapping parts linearly
        final_weights = [torch.tensor(t, device="cuda") for t in final_weights_by_words]
        result_weight = final_weights[0]
        for j in range(1, len(final_weights)):
            transition_length = len([w for w in words if j in w["splits"] and j-1 in w["splits"]])
            transition_weight = (torch.arange(start=0, end=transition_length, step=1) / transition_length).to("cuda")
            result_weight[-transition_length:] = result_weight[-transition_length:] * (1-transition_weight) + final_weights[j][:transition_length] * transition_weight
            result_weight = torch.cat([result_weight, final_weights[j][transition_length:]])
        result_weight = result_weight.detach().cpu().numpy()

        # Normalize scores
        result_weight = result_weight - np.min(result_weight)
        result_weight = result_weight / np.max(result_weight)

        visualize_word_importance(list(zip(result_weight, [w["word"] for w in words])))

        return result_weight