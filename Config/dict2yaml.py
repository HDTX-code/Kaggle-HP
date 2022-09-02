import yaml
import json


def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict保存为yaml"""
    with open(save_path, 'w') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))


def read_yaml_to_dict(yaml_path: str, ):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value


if __name__ == '__main__':
    norm_cfg = dict(type='BN', requires_grad=True)
    num_classes = 5
    cfg = dict(
        backbone=dict(
            pre_url=r"D:\edge\convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth",
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
    # 保存yaml
    save_dict_to_yaml(cfg, "config.yaml")
    # 读取yaml
    config_value = read_yaml_to_dict("config.yaml")
    assert config_value == cfg
    print(json.dumps(config_value, sort_keys=False, indent=4))
