import logging

from .func import *
from .quantizer import *
from collections import defaultdict

#todo: take from config.yml..
def create_quan_scheduler():
    quan_scheduler = defaultdict(dict)
    quan_scheduler['act'] = {'mode': 'lsq', 'bit': 8, 'per_channel': False, 'symmetric': False, 'all_positive': True}
    quan_scheduler['weight'] = {'mode': 'lsq', 'bit': 8, 'per_channel': False, 'symmetric': True, 'all_positive': False}
    quan_scheduler['excepts'] = {'conv1': {'act': {'bit': None, 'all_positive': False}, 'weight': {'bit': None}},
                                 'fc': {'act': {'bit': None}, 'weight': {'bit': None}},
                                 'conv_stem': {'act': {'bit': None, 'all_positive': False}, 'weight': {'bit': None}},
                                 'classifier': {'act': {'bit': None}, 'weight': {'bit': None}},
                                 'classifier.1': {'act': {'bit': None}, 'weight': {'bit': None}},
                                'conv_head': {'act': {'bit': None}, 'weight': {'bit': None}},
                                'features.18.0': {'act': {'bit': None}, 'weight': {'bit': None}},
                                'features.0.0': {'act': {'bit': None}, 'weight': {'bit': None}},}
    return quan_scheduler

def quantizer(default_cfg,this_cfg=None,i=None,list_bit=None):
    target_cfg = dict(default_cfg)
    # list_bit = [3,4,16,4,3,4,2,4,4,4,4,4,4,4,3,2,4,4,4] - resnet18 imagenet
    # list_bit = [8,4,8 ,4,4,4,2,4,4,4,4,4,4,4,3,2,4,4,4]
    #list_bit =   [3,2,4,4,4,4,4,3,4,4,3,4,4,4,4,4,4,4,4] - 18 on IN
    # list_bit = [2,4,4,8,3,4,8,2,8,3,3,8,8,3,4,3,8,8,2,8,3,2,8,3,2,4,8,8,2,4,8,8,8,8,
    #             8,8,8,4,8,4,2,4,3,4,2,3,8,8,8,8,8,3] #- resnet50 imagenet
    # list_bit = [8,2,3,8,8,4,4,2,3,8,4,3,8,2,8,2,8,8,2,2,3,2,2,3,3,3,3,8,8,3,3,2,8,
    #             8,3,2,3,2,3,8,2,2,2,3,4,3,2,3,2,3,8,3] #- resnet50 imagenet new
    is_lsq = False
    if this_cfg is not None:
        for k, v in this_cfg.items():
            target_cfg[k] = v

    if target_cfg['bit'] is None:
        q = IdentityQuan
    elif target_cfg['mode'] == 'lsq':
        q = LsqQuan
        is_lsq = True
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['mode'])

    target_cfg.pop('mode')
    if is_lsq and i is not None:
        #print('set bit to ',list_bit[i])
        #print(i)
        target_cfg['bit'] = list_bit[i]
    return q(**target_cfg)


def find_modules_to_quantize(model, quan_scheduler,list_bit):
    replaced_modules = dict()
    i=0
    for name, module in model.named_modules():
        if type(module) in QuanModuleMapping.keys():
            if name in quan_scheduler['excepts']:
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    quan_w_fn=quantizer(quan_scheduler['weight'],
                                        quan_scheduler['excepts'][name]['weight']),
                    quan_a_fn=quantizer(quan_scheduler['act'],
                                        quan_scheduler['excepts'][name]['act'])
                )
            else:
                #print('for name ', name, i)
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    quan_w_fn=quantizer(quan_scheduler['weight'],i=i,list_bit=list_bit),
                    quan_a_fn=quantizer(quan_scheduler['act'],i=i,list_bit=list_bit)
                )
                i = i + 1
        elif name in quan_scheduler['excepts']:
            logging.warning('Cannot find module %s in the model, skip it' % name)
    return replaced_modules


def replace_module_by_names(model, modules_to_replace):
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)

    helper(model)
    return model
