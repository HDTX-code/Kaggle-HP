import torch.nn.functional as F


def binary_cross_entropy(pred, label, class_weight=None):
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    return loss



