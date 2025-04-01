import torch
import torch.nn as nn
from model.HR_Net.seg_hrnet import HighResolutionNet 
# from sryolo_model.SRyolo import Model ## import SR yolo files later

## remove later
hrnet_config = "..\model\HR_Net\seg_hrnet_w48.yaml"

class SharedHead(nn.Module):
    def __init__(self):
        super().__init__()
        if True:
            self.hrnet_backbone = HighResolutionNet(config=hrnet_config)
        else:
            self.yolo_backbone = Model()

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
