import torch
import torch.nn as nn

from models.SRyolo import SRyolo # SUPERYOLO (later)
from ..model.locator import Crowd_locator
from ..model.PBM import BinarizedModule

class VehicleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SRyolo(config_path='cfg/SRyolo_noFocus.yaml')  ## Change later

    def forward(self, features):
        return self.model(features)

class CrowdHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.density_estimator = BinarizedModule()
        self.locator = Crowd_locator()

    def forward(self, features):
        density_map = self.density_estimator(features)
        instance_map, count = self.locator(density_map)
        return {'density_map': density_map, 'instance_map': instance_map, 'count': count}
