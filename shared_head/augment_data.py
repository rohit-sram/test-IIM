# import pandas as pd
# import numpy as np
# from PIL import Image, ImageOps
# from skimage.filters import sobel
# from skimage.measure import shannon_entropy
# from pathlib import Path
# from tqdm import tqdm

# # === Config ===
# ROOT_DIR = Path("C:/Users/rsriram3/Documents/ind_study")
# CSV_PATH_TRAIN = ROOT_DIR / "data/train_split.csv"
# CSV_PATH_TEST = ROOT_DIR / "data/test_split.csv"
# OUTPUT_DIR = ROOT_DIR / "data"
# TRAIN_OUT = OUTPUT_DIR / "train_split_augmented.csv"
# TEST_OUT = OUTPUT_DIR / "test_split_augmented.csv"

# # === Feature extraction using PIL and skimage ===
# def extract_image_features(image_path):
#     try:
#         img = Image.open(image_path).convert("L")  # grayscale
#     except Exception:
#         return np.nan, np.nan, np.nan

#     gray_np = np.array(img)

#     # Brightness: average pixel intensity
#     brightness_mean = np.mean(gray_np)

#     # Edge density: count of non-zero edges from Sobel filter
#     edges = sobel(gray_np)
#     edge_density = np.sum(edges > 0.1) / edges.size  # 0.1 threshold to suppress noise

#     # Entropy: pixel complexity
#     entropy = shannon_entropy(gray_np)

#     return brightness_mean, edge_density, entropy

# # === CSV processor ===
# def process_csv(df, root_dir):
#     brightness_list, edge_list, entropy_list, channel_modes = [], [], [], []

#     print("Extracting image characteristics (PIL-based)...")
#     for _, row in tqdm(df.iterrows(), total=len(df)):
#         image_path = root_dir / row['image']
#         brightness, edges, entropy = extract_image_features(image_path)

#         brightness_list.append(brightness)
#         edge_list.append(edges)
#         entropy_list.append(entropy)

#         filename = row['image'].lower()
#         label = row['label']

#         if label == 1:
#             channel_modes.append('rgb')
#         elif '_ir' in filename:
#             channel_modes.append('ir')
#         elif '_co' in filename:
#             channel_modes.append('rgb')
#         else:
#             channel_modes.append('unknown')

#     df['brightness_mean'] = brightness_list
#     df['edge_density'] = edge_list
#     df['entropy'] = entropy_list
#     df['channel_mode'] = channel_modes

#     return df

# # === Process Train Split ===
# train_df = pd.read_csv(CSV_PATH_TRAIN)
# train_aug = process_csv(train_df, ROOT_DIR)
# train_aug.to_csv(TRAIN_OUT, index=False)
# print(f"✅ Saved: {TRAIN_OUT} with {len(train_aug)} rows and {len(train_aug.columns)} columns.")

# # === Process Test Split ===
# test_df = pd.read_csv(CSV_PATH_TEST)
# test_aug = process_csv(test_df, ROOT_DIR)
# test_aug.to_csv(TEST_OUT, index=False)
# print(f"✅ Saved: {TEST_OUT} with {len(test_aug)} rows and {len(test_aug.columns)} columns.")


"""
NEW
"""

import pandas as pd
import numpy as np
from PIL import Image
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from pathlib import Path
from tqdm import tqdm

# === Config ===
ROOT_DIR = Path("C:/Users/rsriram3/Documents/ind_study")
CSV_PATH_TRAIN = ROOT_DIR / "data/train_split.csv"
CSV_PATH_TEST = ROOT_DIR / "data/test_split.csv"
OUTPUT_DIR = ROOT_DIR / "data"
TRAIN_OUT = OUTPUT_DIR / "train_split_augmented.csv"
TEST_OUT = OUTPUT_DIR / "test_split_augmented.csv"

# === Feature extraction using PIL and skimage ===
def extract_image_features(image_path):
    try:
        if not image_path.exists() or not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            print(f"❌ Skipping invalid file: {image_path}")
            return np.nan, np.nan, np.nan

        img = Image.open(image_path).convert("L")
        gray_np = np.array(img)

        brightness_mean = np.mean(gray_np)
        edge_density = np.sum(sobel(gray_np) > 0.1) / gray_np.size
        entropy = shannon_entropy(gray_np)

        return brightness_mean, edge_density, entropy
    except Exception as e:
        print(f"⚠️ Failed to process {image_path}: {e}")
        return np.nan, np.nan, np.nan

# === CSV processor ===
def process_csv(df, root_dir):
    brightness_list, edge_list, entropy_list, channel_modes = [], [], [], []
    missing_rows = []

    print("Extracting image characteristics (PIL-based)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = root_dir / row['image']
        brightness, edges, entropy = extract_image_features(image_path)

        brightness_list.append(brightness)
        edge_list.append(edges)
        entropy_list.append(entropy)

        filename = row['image'].lower()
        label = row['label']

        if label == 1:
            channel_modes.append('rgb')
        elif '_ir' in filename:
            channel_modes.append('ir')
        elif '_co' in filename:
            channel_modes.append('rgb')
        else:
            channel_modes.append('unknown')

        if np.isnan(brightness) or np.isnan(edges) or np.isnan(entropy):
            missing_rows.append(row['image'])

    df['brightness_mean'] = brightness_list
    df['edge_density'] = edge_list
    df['entropy'] = entropy_list
    df['channel_mode'] = channel_modes

    df_clean = df.dropna(subset=['brightness_mean', 'edge_density', 'entropy'])

    if missing_rows:
        print(f"⚠️ Dropping {len(missing_rows)} rows with missing values. See 'missing_image_rows.csv' for details.")
        pd.DataFrame({'missing_images': missing_rows}).to_csv(OUTPUT_DIR / "missing_image_rows.csv", index=False)

    return df_clean

# === Process Train Split ===
train_df = pd.read_csv(CSV_PATH_TRAIN)
train_aug = process_csv(train_df, ROOT_DIR)
train_aug.to_csv(TRAIN_OUT, index=False)
print(f"✅ Saved: {TRAIN_OUT} with {len(train_aug)} rows and {len(train_aug.columns)} columns.")

# === Process Test Split ===
test_df = pd.read_csv(CSV_PATH_TEST)
test_aug = process_csv(test_df, ROOT_DIR)
test_aug.to_csv(TEST_OUT, index=False)
print(f"✅ Saved: {TEST_OUT} with {len(test_aug)} rows and {len(test_aug.columns)} columns.")