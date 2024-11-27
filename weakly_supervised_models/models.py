import torch
import numpy as np
import copy
import os
import random
from transformers import AutoTokenizer, BertModel
from auxiliary.loss_fn import categorical_cross_entropy_with_logits, binary_cross_entropy_with_logits
from settings import Config

class ClfModel(torch.nn.Module):

    def __init__(self, model, dense_layer):
        super(ClfModel, self).__init__()
        self.model = model
        self.dense_layer = dense_layer

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)["last_hidden_state"]
        clf_out = self.dense_layer(model_output[:, 0])
        return clf_out

class Brinner_2024(torch.nn.Module):

    def __init__(self, model_checkpoint, n_class_labels, **kwargs):
        super(Brinner_2024, self).__init__()
        self.model = BertModel.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

        # Dense layers applied to cls for classification and to word embeddings to predict w and sigma
        self.dense_1 = torch.nn.Linear(768, n_class_labels)
        self.dense_2 = torch.nn.Linear(768, 2*n_class_labels)

        if Config.output_fn == "sigmoid":
            self.loss_fn = binary_cross_entropy_with_logits
        else:
            self.loss_fn = categorical_cross_entropy_with_logits

        self.class_weights = None
        self.n_warmup_epochs = 3 # Warmup: Train only the classifier for 3 epochs
        self.n_class_labels = n_class_labels

        self.evals_without_improvement = 0
        self.max_eval_score = 0
        self.parameters_updated()

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def forward(self, model_inputs, **kwargs):
        # Handle either input embeddings or input ids
        if "inputs_embeds" in model_inputs:
            bert_out = self.model(inputs_embeds=model_inputs["inputs_embeds"])["last_hidden_state"]
        else:
            bert_out = self.model(**model_inputs)["last_hidden_state"]

        text_clf_out = self.dense_1(bert_out[:, 0])
        span_pred_out = self.dense_2(bert_out[:, 1:-1])
        span_pred_out = span_pred_out.reshape(*span_pred_out.shape[:2], self.n_class_labels, 2)

        # If the mask calculation from the network outputs is not needed, it can be omitted.
        if "omit_mask_calc" in kwargs and kwargs["omit_mask_calc"]:
            return text_clf_out, span_pred_out, None

        mask, regularizer_statistics = self.calculate_mask(span_pred_out, model_inputs["input_ids"])
        return text_clf_out, mask, regularizer_statistics

    def frozen_forward(self, inputs_embeds):
        bert_out = self.frozen_model(inputs_embeds=inputs_embeds)["last_hidden_state"]
        text_clf_out = self.frozen_dense_1(bert_out[:, 0])
        return text_clf_out

    # Calculate loss only for classification part of the network
    def calculate_classification_loss(self, model_output, gt_class_labels):
        return self.loss_fn(model_output[0], gt_class_labels[:model_output[0].shape[0]], weights=self.class_weights).mean()

    # Calculate complete loss for a given training sample
    def calculate_training_loss(self, input_batch, gt_class_labels, epoch):
        self.frozen_model.eval()
        self.model.train()

        # We need different representations of the labels
        gt_class_labels = gt_class_labels.unsqueeze(0).repeat(input_batch["input_ids"].shape[0], 1)
        class_labels_int = [l for l in range(self.n_class_labels) if int(gt_class_labels[0, l]) == 1]
        individual_class_labels_one_hot = torch.nn.functional.one_hot(torch.tensor(class_labels_int, device="cuda"), num_classes=self.n_class_labels)

        # Create embeddings for the given input ids as well as the "background" embeddings that unimportant tokens are blended towards. Control tokens can not be blended towards 0.
        text_embeddings = self.model.embeddings(input_batch["input_ids"])
        pad_batch = torch.clone(input_batch["input_ids"])
        control_tokens = torch.isin(input_batch["input_ids"], torch.tensor(self.tokenizer.all_special_ids, device="cuda"))
        pad_batch[torch.logical_not(control_tokens)] = self.tokenizer.pad_token_id
        pad_embeddings = self.model.embeddings(pad_batch)

        # Train model on unperturbed input to learn the standard classification task.
        model_output = self.forward(input_batch)
        loss = 2 * self.calculate_classification_loss(model_output, gt_class_labels)


        average_weights = []
        # Only begin mask training after warmup epochs, so that the classifier has already learned something.
        if epoch >= self.n_warmup_epochs:
            # The individual mask values (all weights) were returned in the forward pass and are now used to create rationaluzed inputs.
            all_weights, regularizer_statistics = model_output[1], model_output[2]
            # Pad the weights to account for control tokens on both sides.
            all_weights = torch.nn.functional.pad(torch.swapaxes(all_weights, 1, 2), (1, 1), value=0)

            # Long texts are split into multiple parts. Loss is calculated for each split individually.
            for split in range(input_batch["input_ids"].shape[0]):
                current_split_embeds = text_embeddings[split].unsqueeze(0)
                current_split_weights = torch.stack([all_weights[split, l] for l in class_labels_int], dim=0).unsqueeze(-1)
                average_weights.append(float(torch.mean(current_split_weights)))

                # Create rationalized inputs
                positive_embeddings = current_split_weights * current_split_embeds + (1 - current_split_weights) * pad_embeddings[split].unsqueeze(0)
                complement_embeddings = (1 - current_split_weights) * current_split_embeds + current_split_weights * pad_embeddings[split].unsqueeze(0)
                input_embeddings = torch.concatenate([positive_embeddings, complement_embeddings], dim=0)

                prediction = self.frozen_forward(inputs_embeds=input_embeddings)
                positive_prediction = prediction[:len(class_labels_int)]
                complement_prediction = prediction[len(class_labels_int):]

                loss += 5 * (-(torch.log(torch.sigmoid(positive_prediction) * 0.9999 + 0.00005) * individual_class_labels_one_hot).sum())
                if Config.output_fn == "softmax":
                    complement_class_probabilities = torch.softmax(complement_prediction, dim=-1)
                    loss += 2.5 * (2 * (torch.relu(complement_class_probabilities - (1/self.n_class_labels)) * individual_class_labels_one_hot).sum())
                else:
                    complement_class_probabilities = torch.sigmoid(complement_prediction)
                    loss += 2.5 * (2 * (torch.relu(complement_class_probabilities - 0.4) * individual_class_labels_one_hot).sum())

            loss += 3 * self.calculate_regularization_loss(regularizer_statistics, class_labels_int)

        return {"loss": loss,
                "average_weights": float(np.mean(average_weights)) if len(average_weights) > 0 else None,
                "prediction_precision": [int(int(torch.argmax(model_output[0][split])) in class_labels_int) for split in range(input_batch["input_ids"].shape[0])]}

    # Calculate mask from neural network output using MaRC parameterization. The model outputs w and sigma for each word, which are used to calculate masks in the range [0, 1].
    def calculate_mask(self, model_output, input_ids):
        control_tokens = torch.isin(input_ids, torch.tensor(self.tokenizer.all_special_ids, device="cuda")).np()
        end_indices = [1+np.argmax(x[1:]) for x in control_tokens]

        all_individual_sigmas = []
        all_individual_mask_values = []
        result_weights = []
        for split in range(len(input_ids)):
            current_num_words = end_indices[split] - 1
            sigmas_tensor = torch.exp(model_output[split, :current_num_words, :, 0])
            all_individual_sigmas.append(sigmas_tensor)
            weights_tensor = model_output[split, :current_num_words, :, 1]

            distance_values = (torch.arange(start=0, end=current_num_words, step=1).repeat((current_num_words, 1)) -
                               torch.unsqueeze(torch.arange(start=0, end=current_num_words, step=1), -1)).to("cuda").unsqueeze(-1)
            distance_values = torch.square(distance_values) / torch.unsqueeze(sigmas_tensor, dim=1)
            mask_values = torch.sigmoid((torch.exp(-distance_values) * torch.unsqueeze(weights_tensor, dim=0)).sum(dim=1))
            all_individual_mask_values.append(mask_values)

            result_weights.append(torch.nn.functional.pad(mask_values, (0, 0, 0, input_ids.shape[1] - end_indices[split] - 1), value=0))

        return torch.stack(result_weights, dim=0), (all_individual_sigmas, all_individual_mask_values)

    # Was used to randomly blend parts of the input tokens towards PAD tokens, to train the model to get used to this kind of input. We ended up not using this.
    def create_randomly_masked_sample(self, text_embeddings, pad_embeddings):
        random_tensor = torch.rand((1, 1, text_embeddings.shape[1]), device="cuda")
        padded_tensor = torch.nn.functional.pad(random_tensor, (2, 2), mode='replicate')
        kernel = torch.tensor([1, 2, 4, 2, 1], dtype=torch.float32, device="cuda") / 10
        kernel = kernel.view(1, 1, -1)
        smoothed_tensor = torch.nn.functional.conv1d(padded_tensor, kernel, stride=1, groups=1).reshape(1, -1, 1)
        randomized_text_embeddings = smoothed_tensor * text_embeddings + (1 - smoothed_tensor) * pad_embeddings
        return randomized_text_embeddings

    def calculate_regularization_loss(self, regularizer_statistics, class_labels_int):
        average_weights_correct = [torch.stack([torch.mean(x[:, j]) for j in class_labels_int], dim=0) for x in regularizer_statistics[1]]
        average_weights_incorrect = [torch.stack([torch.mean(x[:, j]) for j in range(self.n_class_labels) if j not in class_labels_int], dim=0) for x in regularizer_statistics[1]]

        # L2 Weight regularizer
        loss = torch.mean(torch.stack([torch.sum(torch.square(x)) for x in average_weights_correct], dim=0)) * 0.2
        loss += torch.mean(torch.stack([torch.sum(torch.square(x)) for x in average_weights_incorrect], dim=0)) * 0.05

        # L1 Weight regularizer
        loss += torch.mean(torch.stack([torch.sum(x) for x in average_weights_correct], dim=0)) * 0.001
        loss += torch.mean(torch.stack([torch.sum(x) for x in average_weights_incorrect], dim=0)) * 0.001

        # Sigma regularizer
        average_sigma_diff = [torch.mean(torch.square(x - 3), dim=0) for x in regularizer_statistics[0]]
        loss += torch.mean(torch.stack([torch.sum(x) for x in average_sigma_diff], dim=0)) * 0.02

        # Smoothness regularizer
        loss += torch.stack([torch.square(x[1:] - x[:-1]).sum(0).mean() for x in regularizer_statistics[1]], dim=0).sum() * 0.05

        return loss

    # If the model parameters were updated, the frozen copy of the model used to score the rationales also needs to be updated.
    def parameters_updated(self):
        self.frozen_model = copy.deepcopy(self.model)
        self.frozen_dense_1 = copy.deepcopy(self.dense_1)
        for param in self.frozen_model.parameters():
            param.requires_grad = False
        for param in self.frozen_dense_1.parameters():
            param.requires_grad = False

    def get_clf_model(self):
        return ClfModel(self.model, self.dense_1)


class Lei_2016(torch.nn.Module):

    def __init__(self, model_checkpoint, n_class_labels, **kwargs):
        super(Lei_2016, self).__init__()
        self.weight_factor = kwargs["weight_penalty"] if "weight_penalty" in kwargs else 0.4
        self.tv_factor = kwargs["total_variation_penalty"] if "total_variation_penalty" in kwargs else 0.2
        self.generator = BertModel.from_pretrained(model_checkpoint)
        self.encoder = BertModel.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.dense_generator = torch.nn.Linear(768, 1)
        self.dense_encoder = torch.nn.Linear(768, n_class_labels)

        self.class_weights = None
        self.n_warmup_epochs = 1
        self.n_class_labels = n_class_labels
        self.n_samples_per_step = 8

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights.reshape(1, 1, -1)

    def forward(self, model_inputs, **kwargs):
        generator_out = self.generator(**model_inputs)["last_hidden_state"]
        mask_probabilities = torch.sigmoid(self.dense_generator(generator_out))

        maximum_likelihood_mask = (mask_probabilities > 0.5).squeeze(-1)
        model_inputs["input_ids"] = self.apply_mask(model_inputs["input_ids"], maximum_likelihood_mask)

        encoder_out = self.encoder(**model_inputs)["last_hidden_state"]
        class_logits = self.dense_encoder(encoder_out[:, 0])

        return class_logits, mask_probabilities[:, 1:-1].repeat(1, 1, self.n_class_labels)

    # Calculate complete loss for a given training sample
    def calculate_training_loss(self, input_batch, gt_class_labels, epoch):
        gt_class_labels = gt_class_labels.unsqueeze(0).repeat(input_batch["input_ids"].shape[0], 1)
        class_labels_int = [l for l in range(self.n_class_labels) if int(gt_class_labels[0, l]) == 1]
        if Config.output_fn == "softmax":
            gt_class_labels = gt_class_labels / torch.sum(gt_class_labels, dim=-1, keepdim=True)

        # If sample is split into multiple parts, train on only on of them (due to memory limitations).
        # You may remove this block if wanted.
        if input_batch["input_ids"].shape[0] > 1:
            selection = random.randrange(0, input_batch["input_ids"].shape[0])
            for key in input_batch:
                input_batch[key] = input_batch[key][selection].unsqueeze(0)
            gt_class_labels = gt_class_labels[selection].unsqueeze(0)

        generator_out = self.generator(**input_batch)["last_hidden_state"]
        mask_probabilities = torch.sigmoid(self.dense_generator(generator_out)).squeeze(-1)
        if epoch < self.n_warmup_epochs:
            # Increase sampling probabilities during warmup to help encoder learn class indicators from more unmasked features
            # before adapting the generator
            mask_probabilities = mask_probabilities + 0.2

        mask_samples = [torch.rand_like(mask_probabilities) < mask_probabilities for _ in range(self.n_samples_per_step)]
        altered_inputs = [self.apply_mask(input_batch["input_ids"], m) for m in mask_samples]
        altered_inputs = torch.stack(altered_inputs, dim=0).reshape(-1, mask_probabilities.shape[-1])

        if Config.output_fn == "softmax":
            label_predictions = torch.softmax(self.dense_encoder(self.encoder(input_ids=altered_inputs)["last_hidden_state"][:, 0]), dim=-1)
        else:
            label_predictions = torch.sigmoid(self.dense_encoder(self.encoder(input_ids=altered_inputs)["last_hidden_state"][:, 0]))
        label_predictions = label_predictions.reshape(self.n_samples_per_step, -1, self.n_class_labels)
        label_diff = torch.square(label_predictions - gt_class_labels.unsqueeze(0))
        label_loss = label_diff.sum(-1)
        encoder_loss = (label_loss * (gt_class_labels.unsqueeze(0) * self.class_weights + (1-gt_class_labels).unsqueeze(0) * (self.class_weights))).sum(-1).mean(0).sum()

        if epoch >= self.n_warmup_epochs:
            mask_samples = [m.to(torch.float32) for m in mask_samples]
            weight_penalty = torch.abs(torch.stack([m.mean(-1) for m in mask_samples], dim=0) - 0.3)
            total_variation_penalty = torch.stack([torch.abs(m[:, 2:-1] - m[:, 1:-2]).mean(-1) for m in mask_samples], dim=0)
            mask_costs = label_loss.detach() + weight_penalty * self.weight_factor + total_variation_penalty * self.tv_factor

            # Real likelihoods would be .sum(-1), but we find it more sensible to normalize w.r.t. input length.
            mask_log_likelihoods = torch.stack([torch.log(mask_probabilities * m + (1-mask_probabilities) * (1-m)).mean(-1) for m in mask_samples], dim=0)

            generator_loss = (mask_costs * mask_log_likelihoods).mean(0).sum() * torch.mean(torch.stack([self.class_weights[0, 0, x] for x in class_labels_int]))
        else:
            generator_loss = 0

        loss = (encoder_loss + generator_loss)
        if Config.output_fn == "softmax":
            precision = [np.mean([int(x in class_labels_int) for x in label_predictions.argmax(-1).flatten().tolist()])]
        else:
            precision = [float(((label_predictions > 0.5).to(torch.float32) * gt_class_labels.unsqueeze(0)).sum() / torch.clip(torch.sum(label_predictions > 0.5), min=1))]
        return {"loss": loss,
                "average_weights": float(torch.mean(mask_probabilities)),
                "prediction_precision": precision}

    # Calculate mask from neural network output using MaRC parameterization
    def apply_mask(self, input_ids, mask):
        copied_input_ids = torch.clone(input_ids)
        control_tokens = torch.isin(input_ids, torch.tensor(self.tokenizer.all_special_ids, device="cuda"))
        copied_input_ids[torch.logical_not(mask) & torch.logical_not(control_tokens)] = self.tokenizer.pad_token_id
        return copied_input_ids

    def parameters_updated(self):
        pass

    def get_clf_model(self):
        return ClfModel(self.encoder, self.dense_encoder)


class Yu_2019(torch.nn.Module):

    def __init__(self, model_checkpoint, n_class_labels, **kwargs):
        super(Yu_2019, self).__init__()
        self.weight_factor = kwargs["weight_penalty"] if "weight_penalty" in kwargs else 0.6
        self.tv_factor = kwargs["total_variation_penalty"] if "total_variation_penalty" in kwargs else 0.3
        self.generator = BertModel.from_pretrained(model_checkpoint)
        self.predictor = BertModel.from_pretrained(model_checkpoint)
        self.complement_predictor = BertModel.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.dense_generator = torch.nn.Linear(768, 1)
        self.dense_predictor = torch.nn.Linear(768, n_class_labels)
        self.dense_complement_predictor = torch.nn.Linear(768, n_class_labels)

        if Config.output_fn == "softmax":
            self.loss_fn = categorical_cross_entropy_with_logits
        else:
            self.loss_fn = binary_cross_entropy_with_logits

        self.class_weights = None
        self.n_warmup_epochs = 1
        self.n_class_labels = n_class_labels
        self.n_samples_per_step = 8

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights.reshape(1, 1, -1)

    def forward(self, model_inputs, **kwargs):
        generator_out = self.generator(**model_inputs)["last_hidden_state"]
        mask_probabilities = torch.sigmoid(self.dense_generator(generator_out))

        maximum_likelihood_mask = (mask_probabilities > 0.5).squeeze(-1)
        model_inputs["input_ids"] = self.apply_mask(model_inputs["input_ids"], maximum_likelihood_mask)

        predictor_out = self.predictor(**model_inputs)["last_hidden_state"]
        class_logits = self.dense_predictor(predictor_out[:, 0])

        return class_logits, mask_probabilities[:, 1:-1].repeat(1, 1, self.n_class_labels)

    # Calculate complete loss for a given training sample
    def calculate_training_loss(self, input_batch, gt_class_labels, epoch):
        gt_class_labels = gt_class_labels.unsqueeze(0).repeat(input_batch["input_ids"].shape[0], 1)
        class_labels_int = [l for l in range(self.n_class_labels) if int(gt_class_labels[0, l]) == 1]
        if Config.output_fn == "softmax":
            gt_class_labels = gt_class_labels / torch.sum(gt_class_labels, dim=-1, keepdim=True)

        # If sample is split into multiple parts, train on only on of them (due to memory limitations).
        if input_batch["input_ids"].shape[0] > 1:
            selection = random.randrange(0, input_batch["input_ids"].shape[0])
            for key in input_batch:
                input_batch[key] = input_batch[key][selection].unsqueeze(0)
            gt_class_labels = gt_class_labels[selection].unsqueeze(0)

        generator_out = self.generator(**input_batch)["last_hidden_state"]
        mask_probabilities = torch.sigmoid(self.dense_generator(generator_out)).squeeze(-1)
        if epoch < self.n_warmup_epochs:
            # Increase sampling probabilities during warmup to help encoder learn class indicators from more unmasked features
            # before adapting the generator
            mask_probabilities = mask_probabilities + 0.2

        mask_samples = [torch.rand_like(mask_probabilities) < mask_probabilities for _ in range(self.n_samples_per_step)]
        positive_inputs = [self.apply_mask(input_batch["input_ids"], m) for m in mask_samples]
        positive_inputs = torch.stack(positive_inputs, dim=0).reshape(-1, mask_probabilities.shape[-1])
        complement_inputs = [self.apply_mask(input_batch["input_ids"], torch.logical_not(m)) for m in mask_samples]
        complement_inputs = torch.stack(complement_inputs, dim=0).reshape(-1, mask_probabilities.shape[-1])
        if epoch < self.n_warmup_epochs:
            complement_inputs = positive_inputs

        positive_label_predictions = self.dense_predictor(self.predictor(input_ids=positive_inputs)["last_hidden_state"][:, 0])
        positive_label_predictions = positive_label_predictions.reshape(self.n_samples_per_step, -1, self.n_class_labels)
        positive_label_loss = self.loss_fn(positive_label_predictions, gt_class_labels.unsqueeze(0)).reshape(-1, 1)
        weighted_positive_label_loss = self.loss_fn(positive_label_predictions, gt_class_labels.unsqueeze(0), self.class_weights)
        predictor_loss = weighted_positive_label_loss.mean(0).sum()

        complement_label_predictions = self.dense_complement_predictor(self.complement_predictor(input_ids=complement_inputs)["last_hidden_state"][:, 0]).reshape(self.n_samples_per_step, -1, self.n_class_labels)
        if Config.output_fn == "softmax":
            complement_label_probabilities = torch.softmax(complement_label_predictions, dim=-1)
            complement_label_loss = (torch.relu(complement_label_probabilities - (1/self.n_class_labels)) * gt_class_labels.unsqueeze(0)).sum(-1).reshape(-1, 1)
        else:
            complement_label_probabilities = torch.sigmoid(complement_label_predictions)
            complement_label_loss = (torch.relu(complement_label_probabilities - 0.5) * gt_class_labels.unsqueeze(0)).sum(-1).reshape(-1, 1)
        complement_predictor_loss = self.loss_fn(complement_label_predictions, gt_class_labels.unsqueeze(0), self.class_weights).mean(0).sum()

        if epoch >= self.n_warmup_epochs:
            mask_samples = [m.to(torch.float32) for m in mask_samples]

            weight_penalty = torch.square(torch.stack([m.mean(-1) for m in mask_samples], dim=0) - 0.3)
            total_variation_penalty = torch.stack([torch.abs(m[:, 2:-1] - m[:, 1:-2]).mean(-1) for m in mask_samples], dim=0)
            mask_costs = positive_label_loss + complement_label_loss + weight_penalty*self.weight_factor + total_variation_penalty * self.tv_factor

            # Real likelihoods would be .sum(-1), but we find it more sensible to normalize w.r.t. input length.
            mask_log_likelihoods = torch.stack([torch.log(mask_probabilities * m + (1-mask_probabilities) * (1-m)).mean(-1) for m in mask_samples], dim=0)

            generator_loss = (mask_costs.detach() * mask_log_likelihoods).mean(0).sum() * torch.mean(torch.stack([self.class_weights[0, 0, x] for x in class_labels_int]))
        else:
            generator_loss = 0

        loss = (predictor_loss + complement_predictor_loss + generator_loss)
        if Config.output_fn == "softmax":
            precision = [np.mean([int(x in class_labels_int) for x in positive_label_predictions.argmax(-1).flatten().tolist()])]
        else:
            precision = [float(((torch.sigmoid(positive_label_predictions) > 0.5).to(torch.float32) * gt_class_labels.unsqueeze(0)).sum() / torch.clip(torch.sum(torch.sigmoid(positive_label_predictions) > 0.5), min=1))]
        return {"loss": loss,
                "average_weights": float(torch.mean(mask_probabilities)),
                "prediction_precision": precision}

    def apply_mask(self, input_ids, mask):
        copied_input_ids = torch.clone(input_ids)
        control_tokens = torch.isin(input_ids, torch.tensor(self.tokenizer.all_special_ids, device="cuda"))
        copied_input_ids[torch.logical_not(mask) & torch.logical_not(control_tokens)] = self.tokenizer.pad_token_id
        return copied_input_ids

    def parameters_updated(self):
        pass

    def get_clf_model(self):
        return ClfModel(self.predictor, self.dense_predictor)

class Straight_Through_Layer(torch.nn.Module):
    def __init__(self):
        super(Straight_Through_Layer, self).__init__()

    def forward(self, x):
        noise = torch.rand_like(x)
        output = (x > noise).float()

        return output - x.detach() + x

class Chang_2019(torch.nn.Module):

    def __init__(self, model_checkpoint, n_class_labels, **kwargs):
        super(Chang_2019, self).__init__()
        self.weight_factor = kwargs["weight_penalty"] if "weight_penalty" in kwargs else 0.6
        self.tv_factor = kwargs["total_variation_penalty"] if "total_variation_penalty" in kwargs else 0.3
        self.generator = BertModel.from_pretrained(model_checkpoint)
        self.predictor = BertModel.from_pretrained(model_checkpoint)
        self.frozen_predictor = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.dense_generator = torch.nn.Linear(768, n_class_labels)
        self.dense_predictor = torch.nn.Linear(768, n_class_labels)
        self.frozen_dense_predictor = None
        self.sampling = Straight_Through_Layer()

        self.class_weights = None
        self.n_class_labels = n_class_labels
        self.parameters_updated()
        self.n_warmup_epochs = 0

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def forward(self, model_inputs, **kwargs):
        generator_out = self.generator(**model_inputs)["last_hidden_state"]
        mask_probabilities = torch.sigmoid(self.dense_generator(generator_out)).squeeze(-1)

        return torch.zeros(1, self.n_class_labels, device="cuda", dtype=torch.float32), mask_probabilities[:, 1:-1]

    # Calculate complete loss for a given training sample
    def calculate_training_loss(self, input_batch, gt_class_labels, epoch):
        gt_class_labels = gt_class_labels.unsqueeze(0).repeat(input_batch["input_ids"].shape[0], 1)
        class_labels_int = [l for l in range(self.n_class_labels) if int(gt_class_labels[0, l]) == 1]
        gt_class_labels = gt_class_labels / torch.sum(gt_class_labels, dim=-1, keepdim=True)

        # If sample is split into multiple parts, train on only on of them (due to memory limitations).
        if input_batch["input_ids"].shape[0] > 1:
            selection = random.randrange(0, input_batch["input_ids"].shape[0])
            for key in input_batch:
                input_batch[key] = input_batch[key][selection].unsqueeze(0)
            gt_class_labels = gt_class_labels[selection].unsqueeze(0)

        generator_out = self.generator(**input_batch)["last_hidden_state"]
        mask_probabilities = torch.sigmoid(self.dense_generator(generator_out)).squeeze(-1)
        sampled_mask = self.sampling(mask_probabilities)
        sampled_mask = torch.swapaxes(sampled_mask, 0, -1)

        text_embeddings = self.predictor.embeddings(input_batch["input_ids"])
        pad_batch = torch.clone(input_batch["input_ids"])
        control_tokens = torch.isin(input_batch["input_ids"], torch.tensor(self.tokenizer.all_special_ids, device="cuda"))
        pad_batch[torch.logical_not(control_tokens)] = self.tokenizer.pad_token_id
        pad_embeddings = self.predictor.embeddings(pad_batch)

        input = text_embeddings * sampled_mask + pad_embeddings * (1-sampled_mask)

        frozen_factuality_predictions = torch.sigmoid(torch.diagonal(self.frozen_dense_predictor(self.frozen_predictor(inputs_embeds=input)["last_hidden_state"][:, 0])))
        loss = -torch.mean(frozen_factuality_predictions)

        factuality_predictions = torch.sigmoid(torch.diagonal(self.dense_predictor(self.predictor(inputs_embeds=input.detach())["last_hidden_state"][:, 0]))) * 0.9999 + 0.00005
        gt_class_labels = torch.tensor(gt_class_labels, device="cuda", dtype=torch.float32).squeeze()
        loss -= (gt_class_labels * torch.log(factuality_predictions) + (1-gt_class_labels) * torch.log(1-factuality_predictions)).mean()

        relevant_mask_probabilities = mask_probabilities[torch.logical_not(control_tokens)]
        sparsity_regularizer = self.weight_factor * torch.square(torch.mean(relevant_mask_probabilities, dim=0) - 0.3).mean()
        continuity_regularizer = self.tv_factor * torch.mean(torch.abs(relevant_mask_probabilities[1:, :] - relevant_mask_probabilities[:-1, :]))
        loss += sparsity_regularizer + continuity_regularizer

        # Note: This model is not a classifier, so defining precision does not make sense here. To monitor training, we still define a somewhat useful score.
        precision = [float(((factuality_predictions > 0.5).to(torch.float32) * gt_class_labels).sum() / gt_class_labels.sum())]

        return {"loss": loss,
                "average_weights": float(torch.mean(relevant_mask_probabilities)),
                "prediction_precision": precision}


    def parameters_updated(self):
        self.frozen_predictor = copy.deepcopy(self.predictor)
        self.frozen_dense_predictor = copy.deepcopy(self.dense_predictor)
        for param in self.frozen_predictor.parameters():
            param.requires_grad = False
        for param in self.frozen_dense_predictor.parameters():
            param.requires_grad = False

    def get_clf_model(self):
        return None

def load_model(config, load_weights_if_available=True):
    model_type = config["model_type"]
    if model_type == "Brinner_2024":
        model = Brinner_2024(Config.model_checkpoint, Config.num_class_labels, **config)
    elif model_type == "Lei_2016":
        model = Lei_2016(Config.model_checkpoint, Config.num_class_labels, **config)
    elif model_type == "Yu_2019":
        model = Yu_2019(Config.model_checkpoint, Config.num_class_labels, **config)
    elif model_type == "Chang_2019":
        model = Chang_2019(Config.model_checkpoint, Config.num_class_labels, **config)
    else:
        raise Exception("Unknown model type.")

    if load_weights_if_available:
        try:
            model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f"../output_data/saved_models/{Config.dataset_type}/{model_type}/{Config.save_name}_weights.pkl")))
            print("Loaded existing model weights!")
        except:
            print("No model weights found!")
    return model

def save_model(model_type, model):
    os.makedirs(os.path.join(os.path.dirname(__file__), f"../output_data/saved_models/{Config.dataset_type}/{model_type}/"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), f"../output_data/saved_models/{Config.dataset_type}/{model_type}/{Config.save_name}_weights.pkl"))