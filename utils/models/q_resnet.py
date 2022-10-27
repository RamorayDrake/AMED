"""
    Quantized ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

"""
resnet20 of torchcv https://github.com/osmr/imgclsmob/blob/f2993d3ce73a2f7ddba05da3891defb08547d504/pytorch/pytorchcv/models/common.py#L413
resnet18 https://github.com/osmr/imgclsmob/blob/f2993d3ce73a2f7ddba05da3891defb08547d504/pytorch/pytorchcv/models/common.py#L413
"""



import torch
import torch.nn as nn
import copy
from ..quantization_utils.quant_modules import *
# from pytorchcv.models.common import ConvBlock
# from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock
import time
import logging


class Q_ResNet20(nn.Module):
    """
        Quantized ResNet20 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, model):
        super().__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')
        #print('*************************************')
        #print(dir(init_block))

        self.quant_input = QuantAct()

        self.quant_init_block_convbn = QuantBnConv2d()
        #todo: maybe delete
        #if not init_block.conv.use_bn:
        #    init_block.conv.use_bn = True

        self.quant_init_block_convbn.set_param(init_block.conv, init_block.bn)

        self.quant_act_int32 = QuantAct()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 3, 3]

        for stage_num in range(0, 3):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResBlockBn()
                #print('===================')
                #print(f"stage{stage_num + 1}.unit{unit_num + 1}")
                #print('===================')
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=4, stride=1)

        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0, 3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


class Q_ResNet18(nn.Module):
    def __init__(self, model):
        super().__init__()
        #features = getattr(model, 'features')
        init_block = getattr(model, 'conv1')
        init_bn = getattr(model, 'bn1')
        #print(dir(init_block))
        self.quant_input = QuantAct()

        self.quant_init_block_convbn = QuantBnConv2d()
        self.quant_init_block_convbn.set_param(init_block, init_bn)

        self.quant_act_int32 = QuantAct()

        #todo: apply only for imagenet
        #self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [2, 2, 2, 2]
        for layer_num in range(0, 4):
            layer = getattr(model, "layer{}".format(layer_num + 1))
            for unit_num in range(0, self.channel[layer_num]):
                #unit = getattr(stage, "conv{}".format(unit_num + 1))
                unit = layer[unit_num]#getattr(stage, "conv{}".format(unit_num + 1))
                quant_unit = Q_ResBlockBn()
                quant_unit.set_param(unit)
                setattr(self, f"layer{layer_num + 1}.conv{unit_num + 1}", quant_unit)
                #stage[unit_num] = quant_unit
            #print(stage)
        self.final_pool = QuantAveragePool2d(kernel_size=4)

        self.quant_act_output = QuantAct()

        output = getattr(model, 'linear')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)
        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)

        #x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                #HERE
                #print(f"stg {stage_num+1} conv {unit_num+1} {act_scaling_factor}")
                tmp_func = getattr(self, f"layer{stage_num+1}.conv{unit_num+1}")
                #tmp_func = getattr(self, f"layer{stage_num+1}")[unit_num]
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)
                #print(f"stg {stage_num+1} conv {unit_num+1} done")

        x = self.final_pool(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


class Q_ResNet50(nn.Module):
    """
        Quantized ResNet50 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, model):
        super().__init__()

        features = getattr(model, 'features')

        init_block = getattr(features, 'init_block')
        self.quant_input = QuantAct()
        self.quant_init_convbn = QuantBnConv2d()
        self.quant_init_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

        self.quant_act_int32 = QuantAct()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 6, 3]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResUnitBn()
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


class Q_ResNet101(nn.Module):
    """
       Quantized ResNet101 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, model):
        super().__init__()

        features = getattr(model, 'features')

        init_block = getattr(features, 'init_block')
        self.quant_input = QuantAct()
        self.quant_init_convbn = QuantBnConv2d()
        self.quant_init_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

        self.quant_act_int32 = QuantAct()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 23, 3]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResUnitBn()
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


class Q_ResUnitBn(nn.Module):
    """
       Quantized ResNet unit with residual path.
    """
    def __init__(self):
        super(Q_ResUnitBn, self).__init__()

    def set_param(self, unit):
        self.resize_identity = unit.resize_identity

        self.quant_act = QuantAct()

        convbn1 = unit.body.conv1
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)
        self.quant_act1 = QuantAct()

        convbn2 = unit.body.conv2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)
        self.quant_act2 = QuantAct()

        convbn3 = unit.body.conv3
        self.quant_convbn3 = QuantBnConv2d()
        self.quant_convbn3.set_param(convbn3.conv, convbn3.bn)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn3(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None)

        x = nn.ReLU()(x)

        return x, act_scaling_factor


class Q_ResUnitBn_torchvision(nn.Module):
    """
       Quantized ResNet unit with residual path.
    """
    def __init__(self):
        super(Q_ResUnitBn, self).__init__()

    def set_param(self, unit):
        self.resize_identity = unit.shortcut

        self.quant_act = QuantAct()

        convbn1 = unit.body.conv1
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)
        self.quant_act1 = QuantAct()

        convbn2 = unit.body.conv2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)
        self.quant_act2 = QuantAct()

        convbn3 = unit.body.conv3
        self.quant_convbn3 = QuantBnConv2d()
        self.quant_convbn3.set_param(convbn3.conv, convbn3.bn)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn3(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None)

        x = nn.ReLU()(x)

        return x, act_scaling_factor


class Q_ResBlockBn_torchvision(nn.Module):
    """
        Quantized ResNet block with residual path.
    """
    def __init__(self):
        super(Q_ResBlockBn, self).__init__()

    def set_param(self, unit):
        self.resize_identity = unit.is_shortcut

        self.quant_act = QuantAct()

        conv1 = unit.conv1
        bn1= unit.bn1
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(conv1,bn1)

        self.quant_act1 = QuantAct()

        conv2 = unit.conv2
        bn2 = unit.bn2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(conv2, bn2)

        if self.resize_identity:
            shortcut_conv = unit.shortcut[0]
            shortcut_bn = unit.shortcut[1]

            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(shortcut_conv, shortcut_bn)
        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
        #print(f'c1 act sca fact {act_scaling_factor}')
        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)
        #print('c2')
        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)
        #print('id')
        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None)

        x = nn.ReLU()(x)
        return x, act_scaling_factor


class Q_ResBlockBn(nn.Module):
    """
        Quantized ResNet block with residual path.
    """
    def __init__(self):
        super(Q_ResBlockBn, self).__init__()

    def set_param(self, unit):
        self.resize_identity = unit.resize_identity

        self.quant_act = QuantAct()

        convbn1 = unit.body.conv1
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)

        self.quant_act1 = QuantAct()

        convbn2 = unit.body.conv2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None)

        x = nn.ReLU()(x)

        return x, act_scaling_factor



def q_resnet18(model):
    net = Q_ResNet18(model)
    return net


def q_resnet50(model):
    net = Q_ResNet50(model)
    return net


def q_resnet101(model):
    net = Q_ResNet101(model)
    return net

def q_resnet20_cifar10(model):
    net = Q_ResNet20(model)
    return net



def set_q_model(model,args,bit_config,initial=False):
    for name, m in model.named_modules():
        if name in bit_config.keys():
            if initial:
                setattr(m, 'quant_mode', 'symmetric')
                setattr(m, 'bias_bit', args.bias_bit)
                setattr(m, 'quantize_bias', (args.bias_bit != 0))
                setattr(m, 'per_channel', args.channel_wise)
                setattr(m, 'act_percentile', args.act_percentile)
                setattr(m, 'act_range_momentum', args.act_range_momentum)
                setattr(m, 'weight_percentile', args.weight_percentile)
                setattr(m, 'fix_flag', False)
                setattr(m, 'fix_BN', args.fix_BN)
                setattr(m, 'fix_BN_threshold', args.fix_BN_threshold)
                setattr(m, 'training_BN_mode', args.fix_BN)
                setattr(m, 'checkpoint_iter_threshold', args.checkpoint_iter)
                setattr(m, 'save_path', args.save_path)
                setattr(m, 'fixed_point_quantization', args.fixed_point_quantization)

            if type(bit_config[name]) is tuple: #only for inception
                bitwidth = bit_config[name][0]
                if bit_config[name][1] == 'hook':
                    m.register_forward_hook(hook_fn_forward)
                    global hook_keys
                    hook_keys.append(name)
            else:
                bitwidth = bit_config[name]
            #print(f'{name} ******* BW:{bitwidth} *******')
            if hasattr(m, 'activation_bit'):
                setattr(m, 'activation_bit', bitwidth)
                if bitwidth == 4:
                    setattr(m, 'quant_mode', 'asymmetric')
            else:
                setattr(m, 'weight_bit', bitwidth)
    return model


class Q_ResNet18_pytorchcv(nn.Module):
    def __init__(self, model):
        super().__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')
        #print('*************************************')
        #print(dir(init_block))

        self.quant_input = QuantAct()

        self.quant_init_block_convbn = QuantBnConv2d()
        self.quant_init_block_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

        self.quant_act_int32 = QuantAct()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [2, 2, 2, 2]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResBlockBn()
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x
