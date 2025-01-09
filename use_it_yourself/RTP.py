import torch
import copy
import numpy as np


parameters = {"regularization_strength": 1.5,
              "rationale_enforcement": 2.5,
              "complement_rationale_enforcement": 2.5,
              "L1_weight_regularizer_rationale": 0.001,
              "L1_weight_regularizer_complement": 0.001,
              "L2_weight_regularizer_rationale": 0.2,
              "L2_weight_regularizer_complement": 0.05,
              "smoothness_regularizer_sigma": 0.02,
              "smoothness_regularizer_TV": 0.05}

def find_boundary_control_tokens(control_tensor):
    n, m = control_tensor.shape
    start_indices = np.zeros(n, dtype=int)
    end_indices = np.full(n, m, dtype=int)

    for i in range(n):
        row = control_tensor[i]

        if row[0]:
            j = 0
            while j < m and row[j]:
                j += 1
            start_indices[i] = j

        if row[-1]:
            j = m - 1
            while j >= 0 and row[j]:
                j -= 1
            end_indices[i] = j + 1

    return start_indices, end_indices

class RTP(torch.nn.Module):

    def __init__(self, model, tokenizer, embedding_fn, n_class_labels, loss_fn, output_fn="softmax", class_weights=None):
        super(RTP, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.output_fn = output_fn
        self.embedding_fn = embedding_fn

        self.dense_clf = torch.nn.Linear(model.config.hidden_size, n_class_labels).to(self.model.device)
        self.dense_mask = torch.nn.Linear(model.config.hidden_size, 2*n_class_labels).to(self.model.device)

        self.class_weights = class_weights
        self.n_class_labels = n_class_labels

    def forward(self, model_inputs, **kwargs):
        # Handle either input embeddings or input ids
        model_output = self.model(**model_inputs)

        clf_logits = self.dense_clf(model_output["last_hidden_state"][:, 0, :])
        span_pred_out = self.dense_mask(model_output["last_hidden_state"])
        span_pred_out = span_pred_out.reshape(*span_pred_out.shape[:-1], self.n_class_labels, 2)

        # If the mask calculation from the network outputs is not needed, it can be omitted.
        if "omit_mask_calc" in kwargs and kwargs["omit_mask_calc"]:
            return {"logits": clf_logits}

        mask, regularizer_statistics = self.calculate_mask(span_pred_out, model_inputs["input_ids"])
        return {"logits": clf_logits, "importance_scores": mask, "regularizer_statistics": regularizer_statistics}

    def frozen_forward(self, inputs):
        bert_out = self.dense_clf(self.frozen_model(**inputs)["last_hidden_state"][:, 0])
        return bert_out

    # Calculate loss only for classification part of the network
    def calculate_classification_loss(self, model_output, gt_class_labels):
        return self.loss_fn(model_output, gt_class_labels).mean()

    # Calculate complete loss for a given training sample
    def train_step(self, model_inputs, labels, skip_mask_update=False):
        self.parameters_updated()
        self.frozen_model.eval()
        self.model.train()

        # We need different representations of the labels
        class_labels_int = [[l for l in range(self.n_class_labels) if int(labels[input_idx, l]) > 0] for input_idx in range(model_inputs["input_ids"].shape[0])]
        individual_class_labels_one_hot = [torch.nn.functional.one_hot(torch.tensor(class_labels_int[input_idx], device=self.model.device), num_classes=self.n_class_labels) for input_idx in range(model_inputs["input_ids"].shape[0])]

        # Create embeddings for the given input ids as well as the "background" embeddings that unimportant tokens are blended towards. Control tokens can not be blended towards 0.
        text_embeddings = self.embedding_fn(self.model)(model_inputs["input_ids"])
        pad_batch = torch.clone(model_inputs["input_ids"])
        control_tokens = torch.isin(model_inputs["input_ids"], torch.tensor(self.tokenizer.all_special_ids, device=self.model.device))
        pad_batch[torch.logical_not(control_tokens)] = self.tokenizer.pad_token_id
        pad_embeddings = self.embedding_fn(self.model)(pad_batch)

        # Train model on unperturbed input to learn the standard classification task.
        model_output = self.forward(model_inputs)
        loss = self.calculate_classification_loss(model_output["logits"], labels) * model_inputs["input_ids"].shape[0]

        average_weights = []
        # Only begin mask training after warmup epochs, so that the classifier has already learned something.
        if not skip_mask_update:
            # The individual mask values (all weights) were returned in the forward pass and are now used to create rationaluzed inputs.
            all_weights, regularizer_statistics = model_output["importance_scores"], model_output["regularizer_statistics"]
            all_weights = torch.swapaxes(all_weights, 1, 2)

            # Long texts are split into multiple parts. Loss is calculated for each split individually.
            for sample_idx in range(model_inputs["input_ids"].shape[0]):
                current_split_embeds = text_embeddings[sample_idx].unsqueeze(0)
                current_split_weights = torch.stack([all_weights[sample_idx, l] for l in class_labels_int[sample_idx]], dim=0).unsqueeze(-1)
                average_weights.append(float(torch.mean(current_split_weights)))

                # Create rationalized inputs
                positive_embeddings = current_split_weights * current_split_embeds + (1 - current_split_weights) * pad_embeddings[sample_idx].unsqueeze(0)
                complement_embeddings = (1 - current_split_weights) * current_split_embeds + current_split_weights * pad_embeddings[sample_idx].unsqueeze(0)
                input_embeddings = torch.concatenate([positive_embeddings, complement_embeddings], dim=0)

                prediction = self.frozen_forward({"inputs_embeds": input_embeddings})
                positive_prediction = prediction[:len(class_labels_int[sample_idx])]
                complement_prediction = prediction[len(class_labels_int[sample_idx]):]

                loss += parameters["rationale_enforcement"] * (-(torch.log(torch.sigmoid(positive_prediction) * 0.9999 + 0.00005) * individual_class_labels_one_hot[sample_idx]).sum())

                if self.output_fn == "softmax":
                    complement_class_probabilities = torch.softmax(complement_prediction, dim=-1)
                    loss += parameters["complement_rationale_enforcement"] * ((torch.relu(complement_class_probabilities - (1/self.n_class_labels)) * individual_class_labels_one_hot[sample_idx]).sum())
                else:
                    complement_class_probabilities = torch.sigmoid(complement_prediction)
                    loss += parameters["complement_rationale_enforcement"] * ((torch.relu(complement_class_probabilities - 0.5) * individual_class_labels_one_hot[sample_idx]).sum())

                loss += parameters["regularization_strength"] * self.calculate_regularization_loss((regularizer_statistics[0][sample_idx], regularizer_statistics[1][sample_idx]), class_labels_int[sample_idx])

        return loss / model_inputs["input_ids"].shape[0]

    # Calculate mask from neural network output using MaRC parameterization. The model outputs w and sigma for each word, which are used to calculate masks in the range [0, 1].
    def calculate_mask(self, model_output, input_ids):
        control_tokens = torch.isin(input_ids, torch.tensor(self.tokenizer.all_special_ids, device=self.model.device)).detach().cpu().numpy()
        start_boundaries, end_boundaries = find_boundary_control_tokens(control_tokens)

        all_individual_sigmas = []
        all_individual_mask_values = []
        result_weights = []
        for sample_idx in range(len(input_ids)):
            sample_mask = model_output[sample_idx, start_boundaries[sample_idx]:end_boundaries[sample_idx]]
            current_num_words = sample_mask.shape[0]
            sigmas_tensor = torch.exp(sample_mask[:, :, 0])
            all_individual_sigmas.append(sigmas_tensor)
            weights_tensor = sample_mask[:, :, 1]

            distance_values = (torch.arange(start=0, end=current_num_words, step=1).repeat((current_num_words, 1)) -
                               torch.unsqueeze(torch.arange(start=0, end=current_num_words, step=1), -1)).to(self.model.device).unsqueeze(-1)
            distance_values = torch.square(distance_values) / torch.unsqueeze(sigmas_tensor, dim=1)
            mask_values = torch.sigmoid((torch.exp(-distance_values) * torch.unsqueeze(weights_tensor, dim=0)).sum(dim=1))
            all_individual_mask_values.append(mask_values)

            result_weights.append(torch.nn.functional.pad(mask_values, (0, 0, start_boundaries[sample_idx], model_output[sample_idx].shape[0] - end_boundaries[sample_idx]), value=0))

        return torch.stack(result_weights, dim=0), (all_individual_sigmas, all_individual_mask_values)

    def calculate_regularization_loss(self, regularizer_statistics, class_labels_int):
        average_weights_correct = [torch.stack([torch.mean(x[:, j]) for j in class_labels_int], dim=0) for x in [regularizer_statistics[1]]]
        average_weights_incorrect = [torch.stack([torch.mean(x[:, j]) for j in range(self.n_class_labels) if j not in class_labels_int], dim=0) for x in [regularizer_statistics[1]]]

        # L2 Weight regularizer
        loss = torch.mean(torch.stack([torch.sum(torch.square(x)) for x in average_weights_correct], dim=0)) * parameters["L2_weight_regularizer_rationale"]
        loss += torch.mean(torch.stack([torch.sum(torch.square(x)) for x in average_weights_incorrect], dim=0)) * parameters["L2_weight_regularizer_complement"]

        # L1 Weight regularizer
        loss += torch.mean(torch.stack([torch.sum(x) for x in average_weights_correct], dim=0)) * parameters["L1_weight_regularizer_rationale"]
        loss += torch.mean(torch.stack([torch.sum(x) for x in average_weights_incorrect], dim=0)) * parameters["L1_weight_regularizer_complement"]

        # Sigma regularizer
        average_sigma_diff = [torch.mean(torch.square(x - 3), dim=0) for x in [regularizer_statistics[0]]]
        loss += torch.mean(torch.stack([torch.sum(x) for x in average_sigma_diff], dim=0)) * parameters["smoothness_regularizer_sigma"]

        # Smoothness regularizer
        loss += torch.stack([torch.square(x[1:] - x[:-1]).sum(0).mean() for x in [regularizer_statistics[1]]], dim=0).sum() * parameters["smoothness_regularizer_TV"]

        return loss

    # If the model parameters were updated, the frozen copy of the model used to score the rationales also needs to be updated.
    def parameters_updated(self):
        self.frozen_model = copy.deepcopy(self.model)
        for param in self.frozen_model.parameters():
            param.requires_grad = False

    def visualize_sample(self, model_inputs):
        from sty import bg

        def colored(r, g, b, text):
            return "\033[38;2;{};{};{}m{}\033[38;2;255;255;255m".format(r, g, b, text)

        def colored_bg(r, g, b, text):
            return bg(r, g, b) + text + bg.rs

        def color_by_importance(importance, gt, word):
            v_1 = int(255 * (1 - importance))
            if gt == 1:
                return colored(40, 181, 40, colored_bg(255, v_1, v_1, word))
            else:
                return colored(0, 0, 0, colored_bg(255, v_1, v_1, word))

        def visualize_word_importance(words):
            if len(words[0]) == 2:
                words = [(w[0], 0, w[1]) for w in words]
            current_line_length = 0
            for i in range(len(words)):
                if current_line_length + len(words[i][2]) + 1 > 180:
                    if i != (len(words) - 1):
                        print(color_by_importance(0, 0, " " * (180 - current_line_length)), end="")
                    print()
                    print(colored(0, 0, 0, color_by_importance(*words[i])), end="")
                    current_line_length = len(words[i][2]) + 1
                else:
                    if i > 0:
                        print(color_by_importance((words[i - 1][0] + words[i][0]) / 2,
                                                  (words[i - 1][1] + words[i][1]) / 2, " "), end="")
                    print(colored(0, 0, 0, color_by_importance(*words[i])), end="")
                    current_line_length += len(words[i][2]) + 1
                if current_line_length > 180:
                    print()
                    current_line_length = 0

            print()

        if model_inputs["input_ids"].shape[0] > 1:
            raise Exception("Please provide a single input sample for visualization!")
        input_ids = model_inputs["input_ids"][0]
        prediction = self.forward(model_inputs)

        real_tokens = np.logical_not(torch.isin(input_ids, torch.tensor(self.tokenizer.all_special_ids, device=self.model.device)).detach().cpu().numpy())
        real_token_list = [self.tokenizer.decode(x) for x in input_ids[real_tokens]]

        for class_idx in range(self.n_class_labels):
            scores = prediction["importance_scores"][0][:, class_idx][real_tokens]
            print(f"Class index: {class_idx}")
            visualize_word_importance(list(zip(scores, real_token_list)))
            print()
        print()