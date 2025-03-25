import cv2
from combined_module import CombinedModule
import importlib
from task_router import TaskRouter

def preprocess_image(image_path):
    # Initialize task router
    router = TaskRouter()

    # Load image for routing
    img = cv2.imread(image_path)
    
    # Determine task type dynamically
    task_type = 'vehicle' if router.route_image(img) == 0 else 'crowd'

    if task_type == 'vehicle':
        # Call prepare_VEDAI for vehicle images
        prepare_VEDAI = importlib.import_module('prepare_VEDAI')
        return prepare_VEDAI.preprocess(image_path)
    elif task_type == 'crowd':
        # Call prepare_SHHB or prepare_SHHA for crowd images
        if 'SHHB' in image_path:
            prepare_SHHB = importlib.import_module('prepare_SHHB')
            return prepare_SHHB.preprocess(image_path)
        elif 'SHHA' in image_path:
            prepare_SHHA = importlib.import_module('prepare_SHHA')
            return prepare_SHHA.preprocess(image_path)

# Initialize system
system = CombinedModule()

# Input image
image = preprocess_image("input.jpg")  # Normalize and resize

# Process image through unified system
output = system(image)

# Print results
if 'detections' in output:  # Vehicle detection results
    print(f"Detected {len(output['detections'])} vehicles.")
else:  # Crowd localization results
    print(f"Detected {output['count']} crowd heads.")
