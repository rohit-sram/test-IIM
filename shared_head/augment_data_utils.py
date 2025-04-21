from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from label_data_utils import compute_image_features

def apply_trivial_augment(image: Image.Image) -> Image.Image:
    transforms = [
        lambda x: x.rotate(random.uniform(-30, 30)),
        lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.6, 1.4)),
        lambda x: ImageOps.mirror(x),
        lambda x: ImageOps.autocontrast(x),
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5))),
        lambda x: ImageOps.solarize(x, threshold=random.randint(64, 192)),
    ]
    random.shuffle(transforms)
    for t in transforms[:random.randint(1, 3)]:
        image = t(image)
    return image

def augment_images_from_csv(csv_path, root_dir, augments_per_image=3):
    df = pd.read_csv(csv_path)
    new_rows = []
    root_dir = Path(root_dir)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Applying TrivialAugment"):
        rel_path = Path(row['image'])
        abs_path = root_dir / rel_path
        if not abs_path.exists():
            continue

        try:
            image = Image.open(abs_path).convert('RGB')
        except:
            continue

        for i in range(augments_per_image):
            augmented = apply_trivial_augment(image)
            aug_name = f"{rel_path.stem}_trivialaug{i}{rel_path.suffix}"
            aug_path = rel_path.parent / aug_name
            abs_aug_path = root_dir / aug_path
            abs_aug_path.parent.mkdir(parents=True, exist_ok=True)
            augmented.save(abs_aug_path)

            brightness, edge_density, entropy = compute_image_features(abs_aug_path)

            new_rows.append({
                'image': str(aug_path).replace("\\", "/"),
                'label': row['label'],
                'brightness': brightness,
                'edge_density': edge_density,
                'entropy': entropy
            })

    return pd.DataFrame(new_rows)
