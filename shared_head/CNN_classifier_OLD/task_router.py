import cv2
import torch
import torch.nn as nn

# def route_image(image_path):
#     img = cv2.imread(image_path)
#     height, width = img.shape[:2]
#     num_channels = img.shape[2] if len(img.shape) == 3 else 1

#     if num_channels == 4:  # RGB + IR image
#         return 'vehicle'
#     elif height % 16 == 0 and width % 16 == 0:  # Divisible by 16 resolution
#         return 'crowd'
#     else:
#         aspect_ratio = width / height
#         new_width = int((width // 16) * 16)
#         new_height = int((height // 16) * 16)
#         resized_img = cv2.resize(img, (new_width, new_height))
#         return 'crowd' if new_width % 16 == 0 else 'vehicle'


# class TaskRouter(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.router = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, kernel_size=3),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(32, 2)  # Outputs probabilities for vehicle or crowd
#         )

#     def forward(self, x):
#         return torch.argmax(self.router(x), dim=1)  # 0 for vehicle, 1 for crowd


class TaskRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 2)  # Outputs probabilities for vehicle or crowd
        )

    def route_image(self, img):
        # Convert image to PyTorch tensor
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
        
        # Route image
        return torch.argmax(self.router(img_tensor), dim=1).item()  # 0 for vehicle, 1 for crowd
