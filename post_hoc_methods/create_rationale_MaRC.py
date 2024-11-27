import warnings
import torch
import numpy as np
from auxiliary.visualize_text import visualize_word_importance
from auxiliary.split_sample import split_sample_and_return_words
from settings import Config

def print_prediction_values(tensor, selection):
    values_np = tensor.detach().cpu().numpy()
    num_splits = int(values_np.shape[0] / 2)
    print("split | value | complement value")
    for i in range(num_splits):
        print(f"  {i}   | {float(values_np[i][selection]):.3f} |     {float(values_np[num_splits + i][selection]):.3f}")

def add_transition_weights_to_words(words, n_splits):
    start_index = 0
    for j in range(n_splits-1):
        transition_length = len([w for w in words if j in w["splits"] and j + 1 in w["splits"]])
        weight_step = 1 / (transition_length+1)
        i = 1
        for w in words[start_index:]:
            if j in w["splits"] and j + 1 in w["splits"]:
                w["weight"] = {j+1: i*weight_step, j: (1-i*weight_step)}
                i += 1
                start_index += 1
            if i == transition_length+1:
                break
    for w in words:
        if "weight" not in w:
            w["weight"] = {i: 1 for i in w["splits"]}

    all_weights = []
    for j in range(n_splits):
        all_weights += [w["weight"][j] for w in words if j in w["weight"]]
    return torch.tensor(np.array(all_weights), dtype=torch.float32, device="cuda")

def exp_inverse_sigmoid(x):
    return torch.exp(torch.log(x/(1-x)))

num_optimization_steps = 700
random_change_percentage = 0.05
weights_loss_factor = 1
sigma_loss_factor = 1.2

# Set multi_label=True if the model outputs multiple logits per sample for multiclass classification (with softmax).
# "models" can be a single model or a list of models (they have to use the same tokenizer). In the latter case, the mask is created to fit all models, which can reduce overfitting to a single model.
def create_rationale_MaRC(models, tokenizer, sample, print_progress=True, gt_indices=None, target_percentage=0.35):
    if type(models) != list:
        models = [models]
    with (warnings.catch_warnings()):
        warnings.filterwarnings("ignore", category=UserWarning)

        for i in range(len(models)):
            models[i].eval()

        words, num_splits, split_token_counts, n_overlaps = split_sample_and_return_words(tokenizer, sample[0], 510)
        transition_weight_tensor = add_transition_weights_to_words(words, num_splits)
        transition_weight_sum = torch.sum(transition_weight_tensor)

        # Lists how many tokens each word has. A token with count -1 will not be optimized (e.g., PAD, CLS).
        split_word_token_counts = [[-1] for _ in range(num_splits)]
        split_words = [list() for _ in range(num_splits)]
        for word in words:
            for split in word["splits"]:
                split_word_token_counts[split].append(word["n_tokens"])
                split_words[split].append(word["word"])

        # Append [PAD] tokens to split_words[split] list to make all splits have the same number of tokens.
        max_split_length = max(split_token_counts.values())
        num_pads_per_split = []
        for i in range(num_splits):
            if (diff := max_split_length - split_token_counts[i]) > 0:
                split_word_token_counts[i] = split_word_token_counts[i] + [-1] * diff
                split_words[i] = split_words[i] + ["[PAD]"] * diff
            num_pads_per_split.append(diff)

        all_word_tokens_counts = []
        for i in range(num_splits):
            split_word_token_counts[i].append(-1)
            all_word_tokens_counts = all_word_tokens_counts + split_word_token_counts[i]

        sample_tokenized = tokenizer(split_words, return_tensors='pt', truncation=False, is_split_into_words=True)
        # Use sequence of PAD token embeddings as uninformative input
        uninformative_input = tokenizer([("[PAD] " * max_split_length)[:-1]]*num_splits, return_tensors='pt', truncation=False)

        embeddings_sample = [m.bert.embeddings(sample_tokenized["input_ids"].to("cuda")) for m in models]
        embeddings_uninformative = [m.bert.embeddings(uninformative_input["input_ids"].to("cuda")) for m in models]
        attention_mask = torch.ones((2 * num_splits, max_split_length + 2)).to("cuda")

        label = sample[1]

        # The parameters to be optimized
        weights = [torch.tensor([1.2], requires_grad=True, device="cuda", dtype=torch.float32) if r != -1 else
                   torch.tensor([-20], requires_grad=False, device="cuda", dtype=torch.float32) for r in all_word_tokens_counts]
        sigmas = [torch.tensor([2.0], requires_grad=True, device="cuda", dtype=torch.float32) if r != -1 else
                  torch.tensor([0.01], requires_grad=False, device="cuda", dtype=torch.float32) for r in all_word_tokens_counts]
        optimizer = torch.optim.AdamW(weights + sigmas, lr=3e-2)

        num_parameters = [len([k for k in r if k != -1]) for r in split_word_token_counts]
        split_word_token_counts = [[x if x != -1 else 1 for x in r] for r in split_word_token_counts]

        #probability_func = (lambda x: torch.softmax(x, dim=-1)) if Config.output_fn == "softmax" or True else torch.sigmoid
        probability_func = (lambda x: torch.softmax(x, dim=-1))
        initial_pred = [torch.mean(probability_func(m(inputs_embeds=e, attention_mask=attention_mask[:num_splits])["logits"]), dim=0) for m, e in zip(models, embeddings_sample)]
        initial_pred = torch.stack(initial_pred, dim=0).mean(0)
        initial_pred = initial_pred[label]

        print(f"Optimizing for label {label}")

        last_mask_mean_at_checkpoint = 1
        last_mask_mean, last_diff = 1, 0
        weight_loss_scaling_factor = 1
        final_weights = {}
        num_optimization_steps = 900
        for i in range(num_optimization_steps):
            mask_tensors = []

            # Store sigmas and mask values for regularization loss calculation. These mask values are different from "mask_tensors", as values are not repeated for words with multiple tokens.
            all_individual_mask_values = []
            all_individual_sigmas = []
            prev_split_end = 0
            for j in range(num_splits):  # Calculate mask values from weights and sigmas
                current_num_words = len(split_word_token_counts[j])

                sigmas_tensor = torch.cat(sigmas[prev_split_end:prev_split_end+current_num_words])
                all_individual_sigmas.append(sigmas_tensor)
                weights_tensor = torch.cat(weights[prev_split_end:prev_split_end+current_num_words])
                prev_split_end = prev_split_end + current_num_words

                distance_values = (torch.arange(start=0, end=current_num_words, step=1).repeat((current_num_words, 1)) -
                                   torch.unsqueeze(torch.arange(start=0, end=current_num_words, step=1), -1)).to("cuda")
                distance_values = torch.square(distance_values) / torch.square(torch.unsqueeze(sigmas_tensor, dim=-1))
                mask_values = torch.sigmoid((torch.exp(-distance_values) * torch.unsqueeze(weights_tensor, dim=-1)).sum(dim=0))
                all_individual_mask_values.append(mask_values[1:-(1+num_pads_per_split[j])])

                mask_values_repeated = [mask_values[k].repeat(split_word_token_counts[j][k]) for k in range(len(split_word_token_counts[j]))]
                mask_values_repeated = torch.unsqueeze(torch.unsqueeze(torch.cat(mask_values_repeated, dim=0), dim=-1), dim=0)
                mask_tensors.append(mask_values_repeated)
            mask_tensors = torch.squeeze(torch.stack(mask_tensors, dim=0), dim=1)

            # Randomly set mask values to 0 or 1
            ones = torch.ones_like(mask_tensors, device="cuda", dtype=torch.float32)
            d_1 = (torch.empty_like(mask_tensors, device="cuda").uniform_() > random_change_percentage).type(torch.float32)  # Select values to set to 0
            d_2 = (torch.empty_like(mask_tensors, device="cuda").uniform_() > random_change_percentage).type(torch.float32)  # Select values to set to 1
            both = (1 - d_1) * (1 - d_2)  # If a word is selected by both, do not change it.
            d_1 = d_1 + both * ones
            d_2 = d_2 + both * ones
            mask_tensors = mask_tensors * d_1 * d_2 + ones * (1 - d_2)

            classification_loss = torch.zeros((1,), device="cuda", dtype=torch.float32)
            all_predictions = []
            for m_idx in range(len(models)):
                masked_embeddings = embeddings_sample[m_idx] * mask_tensors + embeddings_uninformative[m_idx] * (1-mask_tensors)
                complement_masked_embeddings = embeddings_sample[m_idx] * (1 - mask_tensors) + embeddings_uninformative[m_idx] * mask_tensors
                embeddings = torch.cat([masked_embeddings, complement_masked_embeddings], dim=0)
                embeddings = embeddings + torch.empty_like(embeddings, device="cuda").normal_(mean=0.0, std=0.03)  # Add some Gaussian noise to the input


                prediction = models[m_idx](inputs_embeds=embeddings, attention_mask=attention_mask)["logits"]
                all_predictions.append(probability_func(prediction))

                masked_predictions = probability_func(prediction[:num_splits].mean(0)) * 0.9999 + 0.00005
                complement_masked_predictions = probability_func(prediction[num_splits:].mean(0)) * 0.9999 + 0.00005

                masked_loss = -torch.log(masked_predictions[label])
                #complement_masked_loss = -torch.log(1 - complement_masked_predictions[label])
                complement_masked_loss = 2 * torch.relu(complement_masked_predictions[label] - 0.1)

                classification_loss += masked_loss + complement_masked_loss
                classification_loss += 10 * -torch.log(exp_inverse_sigmoid(masked_predictions[label]) /
                                                   (exp_inverse_sigmoid(masked_predictions[label]) + exp_inverse_sigmoid(complement_masked_predictions[label])))

            classification_loss = classification_loss / len(models)
            weights_loss = weights_loss_factor * weight_loss_scaling_factor * torch.square(mask_mean := ((torch.concatenate(all_individual_mask_values, dim=0) * transition_weight_tensor).sum() / transition_weight_sum))
            sigma_loss = sigma_loss_factor * torch.cat([torch.sum(-torch.log(all_individual_sigmas[j]), dim=0, keepdim=True) / num_parameters[j] for j in range(num_splits)], dim=0)
            loss = classification_loss + (sigma_loss + weights_loss).sum()

            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

            if target_percentage is not None:
                m = mask_mean.np()
                diff = last_mask_mean - m
                if m > target_percentage + 0.1:
                    target_diff = (m - target_percentage) / 150
                    diff_delta = diff - target_diff
                    diff_diff = last_diff - diff
                    weight_loss_scaling_factor = 0.8 * weight_loss_scaling_factor + 0.2 * weight_loss_scaling_factor * max(0.7, (1 - 0.9 * np.tanh(((diff_delta / 2) - diff_diff) / 0.002)))
                else:
                    score_scaling = min((prediction[0][label] / initial_pred).np(), (initial_pred * 0.5 / prediction[1][label]).np(), 1.1)
                    weight_scaling = 1 + (m - 0.3)
                    weight_loss_scaling_factor = 0.8 * weight_loss_scaling_factor + 0.2 * weight_loss_scaling_factor * score_scaling * weight_scaling
                weight_loss_scaling_factor = min(weight_loss_scaling_factor, 100)
                last_diff = diff
                last_mask_mean = m
                #print(weight_loss_scaling_factor)

            prediction = torch.stack(all_predictions, dim=0).mean(0)
            if i == 0 and print_progress:
                print("\nInitial values:")
                print_prediction_values(prediction, selection=label)
                print(f"Avg mask values: {mask_mean.detach().cpu().numpy():.3f}")
            if i > 0 and i % 50 == 0:
                mask_mean_np = mask_mean.detach().cpu().numpy()
                if print_progress:
                    print(f"\nIteration {i}")
                    print_prediction_values(prediction, selection=label)
                    print(f"Avg mask values: {mask_mean.detach().cpu().numpy():.3f}")
                for j in range(num_splits):
                    diff = abs(last_mask_mean_at_checkpoint - mask_mean_np)
                    # Check for stop conditions
                    if (diff < 1 / 200 or (diff < 1 / 80 and mask_mean_np < 0.2)) and (
                            i >= 199 or mask_mean_np < 0.3) and j not in final_weights and mask_mean_np < 0.45:
                        final_weights[j] = all_individual_mask_values[j]
                        if print_progress: print(f"Saved mask for split {j}")
                last_mask_mean_at_checkpoint = mask_mean_np
                if len(final_weights) == num_splits:  # Masks for all splits have been saved
                    break

        # If no stop condition for a split has been reached, take the last mask values as result.
        for j in range(num_splits):
            if j not in final_weights:
                final_weights[j] = all_individual_mask_values[j]

        # If the sample was split into multiple parts, merge individual mask parts by blending overlapping parts linearly
        #result_weight = final_weights[0][:-num_pads_per_split[0]] if num_pads_per_split[0] > 0 else final_weights[0]
        result_weight = final_weights[0]
        for j in range(1, len(final_weights)):
            transition_length = len([w for w in words if j in w["splits"] and j - 1 in w["splits"]])
            transition_weight = (torch.arange(start=0, end=transition_length, step=1) / transition_length).to("cuda")
            #tensor = final_weights[j][:-num_pads_per_split[j]] if num_pads_per_split[j] > 0 else final_weights[j]
            tensor = final_weights[j]
            result_weight[-transition_length:] = result_weight[-transition_length:] * (1 - transition_weight) + tensor[:transition_length] * transition_weight
            result_weight = torch.cat([result_weight, tensor[transition_length:]])

        if result_weight.shape[0] != len(words):
            raise Exception()
        result_weight = result_weight.detach().cpu().numpy()
        try:
            if print_progress:
                if gt_indices is not None:
                    visualize_word_importance(list(zip(result_weight,
                                                       [1 if i in gt_indices else 0 for i in range(len(words))],
                                                       [w["word"] for w in words])))
                else:
                    visualize_word_importance(list(zip(result_weight, [w["word"] for w in words])))
        except:
            return None
        return result_weight
