from ultralytics.nn.modules.pretrained_vit import UNIMultiLayerPatches
import timm
import torch.nn as nn
import torch

uni = UNIMultiLayerPatches(layers=[5, 11], fusion_mode='concat')

dummy_input = torch.randn(2, 3, 224, 224)
output = uni(dummy_input)
for i, o in enumerate(output):
    print(f'Feat index: {i}, shape: {o.shape}')
