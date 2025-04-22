import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from skimage.measure import shannon_entropy
from tqdm import tqdm

# === Feature Computation ===
def compute_edge_density(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.mean(edges > 0)

def compute_image_features(img_path):
    img_array = cv2.imread(str(img_path))
    if img_array is None:
        return None, None, None
    img_array = cv2.resize(img_array, (224, 224))
    brightness = np.mean(img_array)
    edge_density = compute_edge_density(img_array)
    entropy = shannon_entropy(img_array)
    return brightness, edge_density, entropy

def generate_label_csv(csv_path, root_dir):
    df = pd.read_csv(csv_path)
    features = []
    root_dir = Path(root_dir)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing image features"):
        rel_path = Path(row['image'])
        img_path = root_dir / rel_path
        brightness, edge_density, entropy = compute_image_features(img_path)
        features.append([brightness, edge_density, entropy])

    df[['brightness', 'edge_density', 'entropy']] = pd.DataFrame(features, columns=['brightness', 'edge_density', 'entropy'])
    return df
