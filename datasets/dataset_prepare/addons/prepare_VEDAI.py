from PIL import Image
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2 as cv
import torch
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
# from ..functions import euclidean_dist, average_del_min

# Define paths
# VEDAI_1024_ROOT = Path('/Users/rohitsriram/Documents/Higher Education/North Carolina State University/Curriculum/Year 2/independent study/vedai_1024')
VEDAI_1024_ROOT = '/mnt/ncsudrive/r/rsriram3/Documents/ind_study/VEDAI_dataset/VEDAI_1024'
VEDAI_IMAGES_PATH = os.path.join(VEDAI_1024_ROOT, 'processed_images')
# VEDAI_LABELS_PATH = "/Users/rohitsriram/Documents/Higher Education/North Carolina State University/Curriculum/Year 2/independent study/vedai_1024_labels"
VEDAI_LABELS_PATH = os.path.join(VEDAI_1024_ROOT, 'labels')
VEDAI_JSON_PATH = os.path.join(VEDAI_1024_ROOT, 'jsons')
VEDAI_SIZE_MAP_PATH = os.path.join(VEDAI_1024_ROOT, 'size_map')
VEDAI_MASK_PATH = os.path.join(VEDAI_1024_ROOT, 'mask')
IMAGE_SIZE = 1024  # Assuming images are resized to 512x512

dst_imgs_path = os.path.join(VEDAI_1024_ROOT, 'processed_images')

os.makedirs(dst_imgs_path, exist_ok=True)

def resize_vedai_images(src_path, resize_factor=1.0):
    file_list = list(Path(src_path).glob("*_co.png"))

    for img_path in file_list:
        img = Image.open(img_path)
        w, h = img.size
        new_w, new_h = int(w * resize_factor), int(h * resize_factor)
        resized_img = img.resize((new_w, new_h), Image.BILINEAR)

        dst_img_path = os.path.join(dst_imgs_path, img_path.name)
        resized_img.save(dst_img_path.replace('_co.png', '.jpg'), quality=95, format='JPEG')
        print(f"Resized {img_path.name} to {new_w}x{new_h}")

# resize_vedai_images(VEDAI_1024_ROOT, resize_factor=1) # Keeping it 1024x1024
# resize_vedai_images(VEDAI_1024_ROOT / "Images", resize_factor=0.5)  # Downscaling to 512x512


## FILES DELETION SCRIPT !!!

# list_dst_imgs_path = os.listdir(dst_imgs_path)

# for img in list_dst_imgs_path:
#     if img.endswith(".png"):
#         os.remove(os.path.join(dst_imgs_path, img))


# Ensure output directory exists
os.makedirs(VEDAI_JSON_PATH, exist_ok=True)

def create_vedai_json():
    """Generates JSON files for the VEDAI dataset containing vehicle count and center points."""
    
    for label_file in tqdm(os.listdir(VEDAI_LABELS_PATH)):
        if not label_file.endswith(".txt"):
            continue

        # Extract image filename
        img_id = label_file.replace(".txt", ".jpg")
        json_path = os.path.join(VEDAI_JSON_PATH, label_file.replace(".txt", ".json"))

        # Skip if JSON already exists
        if os.path.exists(json_path):
            continue

        # Read label file
        label_filepath = os.path.join(VEDAI_LABELS_PATH, label_file)
        df = pd.read_csv(label_filepath, sep=" ", header=None, names=["class", "x_center", "y_center", "width", "height"])

        # Convert normalized coordinates to pixel values
        df["x_center_pixel"] = (df["x_center"] * IMAGE_SIZE).astype(int)
        df["y_center_pixel"] = (df["y_center"] * IMAGE_SIZE).astype(int)

        # Prepare JSON data
        json_data = {
            "img_id": img_id,
            "vehicle_count": len(df),  # Number of lines = number of vehicles
            "points": df[["x_center_pixel", "y_center_pixel"]].values.tolist()  # List of (x, y) points
        }

        # Save JSON file
        with open(json_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"Saved JSON: {json_path}")


def create_size_maps(src_root, size_map_root):
    """Creates blank size maps with the same dimensions as the input images."""
    
    if not os.path.exists(size_map_root):
        os.makedirs(size_map_root, exist_ok=True)
    
    for fname in tqdm(os.listdir(VEDAI_IMAGES_PATH)):
        img_path = os.path.join(VEDAI_IMAGES_PATH, fname)
        # size_map_path = os.path.join(size_map_root, fname.replace('.jpg', '.png'))
        size_map_path = os.path.join(size_map_root, fname)

        # Skip if already processed
        if os.path.exists(size_map_path):
            continue

        # Load image to get dimensions
        img = cv.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {fname}, skipping.")
            continue
        h, w, _ = img.shape

        # Create blank size map
        # size_map = np.ones((h, w), dtype=np.uint8) * 255  # White mask
        size_map = np.zeros((h, w), dtype=np.uint8)

        # Save the size map
        cv.imwrite(size_map_path, size_map, [cv.IMWRITE_JPEG_QUALITY, 95])
        print(f"Created size map for {fname}")


# Ensure output directory exists
os.makedirs(VEDAI_MASK_PATH, exist_ok=True)

def generate_vedai_masks():
    """Generates binary masks for the VEDAI dataset using vehicle center points."""
    
    for img_name in tqdm(os.listdir(VEDAI_IMAGES_PATH)):
        # img_id = img_name.replace(".jpg", "")
        img_id = img_name.split('.')[0]
        mask_path = os.path.join(VEDAI_MASK_PATH, img_name.replace('.jpg', '_mask.png'))

        # Skip if already processed
        if os.path.exists(mask_path):
            continue

        # Load corresponding JSON file
        json_file = os.path.join(VEDAI_JSON_PATH, img_id + ".json")
        if not os.path.exists(json_file):
            print(f"Warning: JSON file not found for {img_name}, skipping.")
            continue

        with open(json_file, "r") as f:
            ImgInfo = json.load(f)

        # Load image to get dimensions
        img_path = os.path.join(VEDAI_IMAGES_PATH, img_name)
        img = cv.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_name}, skipping.")
            continue
        h, w, _ = img.shape

        # Create blank mask (black background)
        mask_map = np.zeros((h, w), dtype=np.uint8)

        # Mark vehicle centers on the mask
        for (x, y) in ImgInfo["points"]:
            if 0 <= x < w and 0 <= y < h:
                mask_map[y, x] = 255  # White pixel for vehicle center

        # Save the mask as JPG
        cv.imwrite(mask_path, mask_map, [cv.IMWRITE_PNG_COMPRESSION, 9])
        print(f"Created mask for {img_name}")


# def generate_vedai_masks():
#     """Generates binary masks for the VEDAI dataset using vehicle center points and saves box visualizations."""
    
#     # Create box_vis directory if it doesn't exist
#     box_vis_path = os.path.join(os.path.dirname(VEDAI_1024_ROOT), 'box_vis')
#     if not os.path.exists(box_vis_path):
#         os.makedirs(box_vis_path)

#     for img_name in tqdm(os.listdir(VEDAI_IMAGES_PATH)):
#         img_id = img_name.split('.')[0]
#         mask_path = os.path.join(VEDAI_MASK_PATH, img_id + '_mask.png')
#         box_vis_path = os.path.join(box_vis_path, img_id + '_box_vis.jpg')

#         # Skip if already processed
#         if os.path.exists(mask_path) and os.path.exists(box_vis_path):
#             continue

#         # Load corresponding JSON file
#         json_file = os.path.join(VEDAI_JSON_PATH, img_id + ".json")
#         if not os.path.exists(json_file):
#             print(f"Warning: JSON file not found for {img_name}, skipping.")
#             continue

#         with open(json_file, "r") as f:
#             ImgInfo = json.load(f)

#         # Load image
#         img_path = os.path.join(VEDAI_IMAGES_PATH, img_name)
#         img = cv.imread(img_path)
#         if img is None:
#             print(f"Warning: Could not read {img_name}, skipping.")
#             continue
#         h, w, _ = img.shape

#         # Create blank mask (black background)
#         mask_map = np.zeros((h, w), dtype=np.uint8)

#         # Prepare for visualization
#         plt.figure(figsize=(12, 8))
#         plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

#         # Mark vehicle centers on the mask and draw boxes for visualization
#         for (x, y), (x1, y1, x2, y2) in zip(ImgInfo["points"], ImgInfo["boxes"]):
#             if 0 <= x < w and 0 <= y < h:
#                 mask_map[int(y), int(x)] = 255  # White pixel for vehicle center
                
#                 # Draw rectangle for box visualization
#                 rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r', linewidth=2)
#                 plt.gca().add_patch(rect)
#                 plt.plot(x, y, 'go', markersize=5)  # Green dot for center

#         # Save the mask as PNG
#         cv.imwrite(mask_path, mask_map, [cv.IMWRITE_PNG_COMPRESSION, 9])

#         # Save box visualization
#         plt.axis('off')
#         plt.tight_layout()
#         plt.savefig(box_vis_path, bbox_inches='tight', pad_inches=0, dpi=300)
#         plt.close()

#         print(f"Created mask and box visualization for {img_name}")


# Run the resizing function
resize_vedai_images(os.path.join(VEDAI_1024_ROOT, 'images'))
# Run the JSON creation function
create_vedai_json()
create_size_maps(VEDAI_1024_ROOT, os.path.join(VEDAI_1024_ROOT, 'size_map'))
generate_vedai_masks()

# sys.path.append(str(Path(__file__).parent.parent))
# from scale_map import main