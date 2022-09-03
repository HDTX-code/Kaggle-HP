import torch

from mmcv.runner import DefaultOptimizerConstructor, get_dist_info

from Model import test

base_lr = 1
decay_rate = 2
base_wd = 0.05
weight_decay = 0.05

expected_stage_wise_lr_wd_convnext = [{
    'weight_decay': 0.0,
    'lr_scale': 128
}, {
    'weight_decay': 0.0,
    'lr_scale': 1
}, {
    'weight_decay': 0.05,
    'lr_scale': 64
}, {
    'weight_decay': 0.0,
    'lr_scale': 64
}, {
    'weight_decay': 0.05,
    'lr_scale': 32
}, {
    'weight_decay': 0.0,
    'lr_scale': 32
}, {
    'weight_decay': 0.05,
    'lr_scale': 16
}, {
    'weight_decay': 0.0,
    'lr_scale': 16
}, {
    'weight_decay': 0.05,
    'lr_scale': 8
}, {
    'weight_decay': 0.0,
    'lr_scale': 8
}, {
    'weight_decay': 0.05,
    'lr_scale': 128
}, {
    'weight_decay': 0.05,
    'lr_scale': 1
}]

expected_layer_wise_lr_wd_convnext = [{
    'weight_decay': 0.0,
    'lr_scale': 128
}, {
    'weight_decay': 0.0,
    'lr_scale': 1
}, {
    'weight_decay': 0.05,
    'lr_scale': 64
}, {
    'weight_decay': 0.0,
    'lr_scale': 64
}, {
    'weight_decay': 0.05,
    'lr_scale': 32
}, {
    'weight_decay': 0.0,
    'lr_scale': 32
}, {
    'weight_decay': 0.05,
    'lr_scale': 16
}, {
    'weight_decay': 0.0,
    'lr_scale': 16
}, {
    'weight_decay': 0.05,
    'lr_scale': 2
}, {
    'weight_decay': 0.0,
    'lr_scale': 2
}, {
    'weight_decay': 0.05,
    'lr_scale': 128
}, {
    'weight_decay': 0.05,
    'lr_scale': 1
}]


def get_layer_id_for_vit(var_name, max_layer_id):
    """Get the layer id to set the different learning rates.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.

    Returns:
        int: Returns the layer id of the key.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.layers'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return max_layer_id - 1


def get_layer_id_for_convnext(var_name, max_layer_id):
    """Get the layer id to set the different learning rates in ``layer_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_layer_id (int): Maximum number of backbone layers.

    Returns:
        int: The id number corresponding to different learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    else:
        return max_layer_id + 1


def get_stage_id_for_convnext(var_name, max_stage_id):
    """Get the stage id to set the different learning rates in ``stage_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_stage_id (int): Maximum number of backbone layers.

    Returns:
        int: The id number corresponding to different learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        return 0
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        return stage_id + 1
    else:
        return max_stage_id - 1


class LearningRateDecayOptimizerConstructor(DefaultOptimizerConstructor):
    """Different learning rates are set for different layers of backbone.

    Note: Currently, this optimizer constructor is built for ConvNeXt,
    BEiT and MAE.
    """

    def add_params(self, params, module, **kwargs):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """

        parameter_groups = {}
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        weight_decay = self.base_wd
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
            if 'layer_wise' in decay_type:
                if 'ConvNeXt' in module.backbone.__class__.__name__:
                    layer_id = get_layer_id_for_convnext(
                        name, self.paramwise_cfg.get('num_layers'))
                elif 'BEiT' in module.backbone.__class__.__name__ or \
                        'MAE' in module.backbone.__class__.__name__:
                    layer_id = get_layer_id_for_vit(name, num_layers)
                else:
                    raise NotImplementedError()
            elif decay_type == 'stage_wise':
                if 'ConvNeXt' in module.backbone.__class__.__name__:
                    layer_id = get_stage_id_for_convnext(name, num_layers)
                else:
                    raise NotImplementedError()
            group_name = f'layer_{layer_id}_{group_name}'

            if group_name not in parameter_groups:
                scale = decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
        params.extend(parameter_groups.values())


def check_optimizer_lr_wd(optimizer, gt_lr_wd):
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['weight_decay'] == base_wd
    # param_groups = optimizer.param_groups
    # assert len(param_groups) == len(gt_lr_wd)
    # for i, param_dict in enumerate(param_groups):
    #     assert param_dict['weight_decay'] == gt_lr_wd[i]['weight_decay']
    #     assert param_dict['lr_scale'] == gt_lr_wd[i]['lr_scale']
    #     assert param_dict['lr_scale'] == param_dict['lr']


def convnext_learning_rate_decay_optimizer_constructor(model, decay="stagewise decay"):
    # Test lr wd for ConvNeXT
    optimizer_cfg = dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05)
    # stagewise decay
    if decay == "stagewise decay":
        stagewise_paramwise_cfg = dict(
            decay_rate=decay_rate, decay_type='stage_wise', num_layers=6)
        optim_constructor = LearningRateDecayOptimizerConstructor(
            optimizer_cfg, stagewise_paramwise_cfg)
        optimizer = optim_constructor(model)
        check_optimizer_lr_wd(optimizer, expected_stage_wise_lr_wd_convnext)
        return optimizer
    else:
        # layerwise decay
        layerwise_paramwise_cfg = dict(
            decay_rate=decay_rate, decay_type='layer_wise', num_layers=6)
        optim_constructor = LearningRateDecayOptimizerConstructor(
            optimizer_cfg, layerwise_paramwise_cfg)
        optimizer = optim_constructor(model)
        check_optimizer_lr_wd(optimizer, expected_layer_wise_lr_wd_convnext)
        return optimizer


if __name__ == '__main__':
    model = test()
    op = convnext_learning_rate_decay_optimizer_constructor(model)
