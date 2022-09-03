import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.backbone import get_ConvNeXt
from Model.head import get_uper_head, get_fcn_head


def get_backbone(cfg: dict):
    type = cfg.pop('type')
    if type == 'ConvNeXt':
        pre_url = cfg.pop('pre_url')
        backbone = get_ConvNeXt(cfg, pre_url)
    else:
        pre_url = cfg.pop('pre_url')
        backbone = get_ConvNeXt(cfg, pre_url)
    return backbone


# 生成decode_head
def get_decode_head(cfg: dict):
    type = cfg.pop('type')
    if type == 'UPerHead':
        init = cfg.pop('init')
        decode_head = get_uper_head(cfg, init=init)
    else:
        init = cfg.pop('init')
        decode_head = get_uper_head(cfg, init=init)
    return decode_head


# 生成auxiliary_head
def get_auxiliary_head(cfg: dict):
    type = cfg.pop('type')
    if type == 'FCNHead':
        init = cfg.pop('init')
        auxiliary_head = get_fcn_head(cfg, init=init)
    else:
        init = cfg.pop('init')
        auxiliary_head = get_fcn_head(cfg, init=init)
    return auxiliary_head


class MixSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixing = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, size=None):
        if size is None:
            size = [1024, 1024]
        x = self.mixing * F.interpolate(x, size=size, mode='bilinear', align_corners=False) \
            + (1 - self.mixing) * F.interpolate(x, size=size, mode='nearest')
        return x


class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5]
    IMAGE_RGB_STD = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5]

    def __init__(self, ):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('std', torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x


class SegModel(nn.Module):
    def __init__(self, size, cfg: dict):
        super().__init__()
        self.RGB = None
        self.MixDownSample = None
        self.MixUpSample = None
        self.auxiliary_head = None
        self.decode_head = None
        self.backbone = None
        self.size = size
        self.cfg = cfg
        self.get_model()

    def get_model(self):
        assert 'backbone' in self.cfg, print("No Backbone in Config")
        assert 'decode_head' in self.cfg, print("No decode_head in Config")
        self.backbone = get_backbone(self.cfg['backbone'])
        self.decode_head = get_decode_head(self.cfg['decode_head'])
        if 'auxiliary_head' in self.cfg:
            self.auxiliary_head = get_auxiliary_head(self.cfg['auxiliary_head'])
        else:
            self.auxiliary_head = None
        self.MixUpSample = MixSample()
        self.MixDownSample = MixSample()
        self.RGB = RGB()

    def forward(self, x):
        shape = x.shape[2:]
        x = self.MixUpSample(x, self.size)
        x = self.RGB(x)
        x = self.backbone(x)
        output = self.MixDownSample(self.decode_head(x), shape)
        aux_output = self.MixDownSample(self.auxiliary_head(x), shape)
        return output, aux_output


def test():
    norm_cfg = dict(type='BN', requires_grad=True)
    num_classes = 5
    cfg = dict(
        backbone=dict(
            pre_url=r"../checkpoints/pretrain/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth",
            type='ConvNeXt',
            arch='base',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False),
        decode_head=dict(
            init='normal',
            type='UPerHead',
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False),
        auxiliary_head=dict(
            init='normal',
            type='FCNHead',
            in_channels=512,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False)
    )
    model = SegModel([128, 128], cfg).train()
    x = torch.ones([3, 3, 32, 32])
    output, aux_output = model(x)
    print(output.shape)
    print(aux_output.shape)
    return model


if __name__ == '__main__':
    test()
