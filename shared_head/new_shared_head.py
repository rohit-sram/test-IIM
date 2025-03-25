import torch
import torch.nn as nn
from model.HR_Net.seg_hrnet import HighResolutionNet

## remove later
hrnet_config = "..\model\HR_Net\seg_hrnet_w48.yaml"

class SharedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HighResolutionNet(config=hrnet_config)

    def forward(self, x):
        return self.backbone(x)

# from model.HR_Net.seg_hrnet import HRNet
# from model.VGG.VGG16_FPN import VGG16_FPN

# class SharedBackbone(nn.Module):
#     def __init__(self, use_hrnet=True):
#         super().__init__()
#         self.backbone = HRNet() if use_hrnet else VGG16_FPN()

#     def forward(self, x):
#         return self.backbone(x)
