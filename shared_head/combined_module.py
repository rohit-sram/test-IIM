import torch
import torch.nn as nn

from shared_head.new_shared_head import SharedHead
from head import preprocess_image
from task_router import TaskRouter
from task_heads import VehicleHead, CrowdHead

# class UnifiedDetectionSystem(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = SharedBackbone()
#         self.vehicle_head = VehicleHead()
#         self.crowd_head = CrowdHead()

#     def forward(self, image_path):
#         task_type = route_image(image_path)
#         features = self.backbone(image_path)

#         if task_type == 'vehicle':
#             return self.vehicle_head(features)
#         elif task_type == 'crowd':
#             return self.crowd_head(features)

image_path = "input.jpg"  # change later

class CombinedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SharedHead()
        self.router = TaskRouter()
        self.vehicle_head = VehicleHead()
        self.crowd_head = CrowdHead()

    def forward(self, x):
        preprocessed_image = preprocess_image(x) ## might have to remove this
        # features = self.backbone(x)
        features = self.backbone(preprocessed_image)
        task_type = self.router.route_image(preprocessed_image)  # 0 for vehicle, 1 for crowd

        if task_type == 0:  # Vehicle detection
            return self.vehicle_head(features)
        elif task_type == 1:  # Crowd localization
            return self.crowd_head(features)
