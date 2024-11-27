import torch

def categorical_cross_entropy_with_logits(y_pred, y_true, weights=None):
    Z = torch.log(torch.sum(torch.exp(y_pred), dim=-1, keepdim=True))
    log_softmax = y_pred - Z
    if weights is not None:
        log_softmax = log_softmax * weights
    log_softmax_for_correct_labels = torch.sum(log_softmax * y_true, dim=-1) / torch.sum(y_true, dim=-1)
    loss = -log_softmax_for_correct_labels
    return loss

def binary_cross_entropy_with_logits(y_pred, y_true, weights=None):
    y_pred_prob = torch.sigmoid(y_pred) * 0.9999 + 0.00005
    loss = torch.log(y_pred_prob) * y_true  + torch.log(1-y_pred_prob) * (1-y_true)
    if weights is not None:
        loss = loss * (weights * y_true + (1/weights) * (1-y_true))
    return -loss.sum(-1)