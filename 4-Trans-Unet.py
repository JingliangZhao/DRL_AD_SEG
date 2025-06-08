import torch
from torch import nn
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
import torchvision.models as models

from net.vit_seg_modeling2 import VisionTransformer as ViT_seg2
from net.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg



def get_transNet2(n_classes):
    img_size = 512
    vit_patches_size = 16
    vit_name = 'R50-ViT-B_16'

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    
    net = ViT_seg2(config_vit, img_size=img_size, num_classes=n_classes)
    
    return net
