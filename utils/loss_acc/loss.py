import torch
import torch.nn as nn

from utils.loss_acc.bce_loss import binary_cross_entropy
from utils.loss_acc.dice_loss import Mutil_label_dice_loss
from utils.loss_acc.lovasz_loss import lovasz_hinge_flat


class SegLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, output, target):
        output = torch.sigmoid(output)
        loss = {}
        assert 'bce_loss' in self.cfg
        loss['bce_loss'] = self.cfg['bce_loss'] * binary_cross_entropy(output, target, ignore_index=255)
        if 'dice_loss' in self.cfg:
            loss['dice_loss'] = self.cfg['dice_loss'] * Mutil_label_dice_loss(output, target, ignore_index=255)
        if 'lovasz_loss' in self.cfg:
            loss['lovasz_loss'] = self.cfg['lovasz_loss'] * lovasz_hinge_flat(output, target)
        return loss


if __name__ == '__main__':
    target = torch.ones([3, 3, 224, 224])
    cfg = dict(
        bce_loss=0.5,
        dice_loss=0.25
    )
    target[..., :112] = 0
    SegLoss = SegLoss(cfg)
    loss = SegLoss(torch.ones([3, 3, 224, 224], requires_grad=True), target)
    for k,v in loss.items():
        print(k)
        print(v)
