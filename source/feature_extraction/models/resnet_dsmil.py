"""
Source: https://github.com/binli123/dsmil-wsi/blob/master/compute_feats.py
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18_dsmil(weights_path):
    resnet = models.resnet18(
        weights=None,
        norm_layer=nn.InstanceNorm2d
    )
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()

    # state dictionary of the randomly initialised resnet extractor
    state_dict_init = resnet.state_dict()

    # pre-trained weights dictionary
    pretrained_state_dict_weights = torch.load(weights_path, map_location='cpu')
    # pop the last 4 weights: they are the weights and biases from the 2 linear projection SimCLR layers
    for i in range(4):
        pretrained_state_dict_weights.popitem()

    # new state dictionary in which we put pre-trained weights with the keys from the i_classifier
    new_state_dict = OrderedDict()
    for (k_init, v_init), (k_pth, v_pth) in zip(
            state_dict_init.items(), pretrained_state_dict_weights.items()
    ):
        # print(k_init, ' '*(30-len(k_init)), k_pth)
        name = k_init
        new_state_dict[name] = v_pth

    # load new state dictionary
    resnet.load_state_dict(new_state_dict)

    # check that we all the weights match
    for (k_mod, v_mod), (k_pth, v_pth) in zip(
            resnet.state_dict().items(), pretrained_state_dict_weights.items()
    ):
        assert (v_mod == v_pth).all()

    return resnet
