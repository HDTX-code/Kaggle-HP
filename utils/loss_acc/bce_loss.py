import torch
import torch.nn.functional as F
import torch.nn as nn


def binary_cross_entropy(pred, label, class_weight=None, ignore_index: int = 255):
    loss = 0
    bce_loss = nn.BCELoss(weight=class_weight)
    assert pred.shape == label.shape
    for c in range(pred.shape[1]):
        pred_c, label_c = pred[:, c, ...].unsqueeze(1), label[:, c, ...].unsqueeze(1).float()
        ge = torch.where(pred_c == ignore_index, torch.zeros_like(pred_c), torch.ones_like(pred_c))
        pred_c, label_c = ge * pred_c, ge * label_c
        loss += bce_loss(pred_c, label_c)
    return loss
