"""
NEW: STRTIFIED 5-FOLD CROSS VAL.
"""

# 🧩 Setup: Import and Configuration
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from timm import create_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import gc
import time
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import random
import numpy as np
import os

# Constants
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_MIXUP = True
ALPHA_MIXUP = 0.4
TEMPERATURE = 2.0

# Local Paths
ROOT_DIR = Path("C:/Users/rsriram3/Documents/ind_study")
OUTPUT_DIR = ROOT_DIR / "data"
TRAIN_CSV = OUTPUT_DIR / "train_split_augmented.csv"
TEST_CSV = OUTPUT_DIR / "test_split_augmented.csv"
CHECKPOINT_DIR = OUTPUT_DIR / "best_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Definition (with engineered features)
class SharedHeadDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.root_dir / row.image
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"[ Failed to load image: {img_path}, error: {e}")
            return self.__getitem__((idx + 1) % len(self.data))

        if image.mode != 'RGB':
            image = image.convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image)
        label = int(row.label)

        brightness = torch.tensor(row.brightness_mean, dtype=torch.float32).unsqueeze(0)
        edge_density = torch.tensor(row.edge_density, dtype=torch.float32).unsqueeze(0)
        entropy = torch.tensor(row.entropy, dtype=torch.float32).unsqueeze(0)

        channel_mode = row.channel_mode
        if channel_mode == 'rgb':
            channel_tensor = torch.tensor([1.0, 0.0])
        elif channel_mode == 'ir':
            channel_tensor = torch.tensor([0.0, 1.0])
        else:
            channel_tensor = torch.tensor([0.0, 0.0])

        # aux_features = torch.cat([brightness, edge_density, entropy, channel_tensor], dim=0)
        aux_features = torch.cat([brightness, edge_density, entropy], dim=0)

        return image_tensor, aux_features, label


# Collate Function
def collate_fun(batch):
    images, aux_feats, labels = zip(*batch)
    return torch.stack(images), torch.stack(aux_feats), torch.tensor(labels)

# Model Definition
class ExtendedModel(nn.Module):
    # def __init__(self, backbone_name, num_aux_features=5, num_classes=2): ## ORIGINALLY num_features=5
    def __init__(self, backbone_name, num_aux_features=3, num_classes=2):
        super().__init__()
        self.backbone = create_model(backbone_name, pretrained=True, num_classes=0)
        backbone_output_dim = self.backbone.num_features
        self.backbone.reset_classifier(0)
        self.classifier = nn.Sequential(
            nn.Linear(backbone_output_dim + num_aux_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, aux):
        feats = self.backbone(x)
        combined = torch.cat([feats, aux], dim=1)
        return self.classifier(combined)

# 🧪 Evaluation
def evaluate_model(model, dataloader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, aux, labels in dataloader:
            images, aux = images.to(DEVICE), aux.to(DEVICE)
            outputs = model(images, aux)
            preds.extend(torch.argmax(outputs, dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    cm = confusion_matrix(targets, preds)
    return acc, prec, rec, f1, cm

# Plotting Accuracy
def plot_metrics(train_accs, val_accs, model_name, save_path):
    save_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title(f'Training vs Validation Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    save_path = save_path / f'{model_name}_accuracy_plot.png'
    plt.savefig(save_path)
    plt.close()

#  Mixup Training
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)

# Model Training Wrapper
def train_model(model_name, train_csv, test_csv, root_dir, save_path):
    train_ds = SharedHeadDataset(train_csv, root_dir)
    test_ds = SharedHeadDataset(test_csv, root_dir)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fun, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fun, pin_memory=True)

    model = ExtendedModel(model_name).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()

    train_accs, val_accs = [], []
    best_acc = 0.0
    previous_logits = None

    for epoch in range(EPOCHS):
        print(f"\n[ Epoch {epoch+1}/{EPOCHS} ]")
        model.train()
        correct, total = 0, 0
        progress = tqdm(train_loader, desc=f"Training {model_name}", leave=False)

        for images, aux, labels in progress:
            images, aux, labels = images.to(DEVICE), aux.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if USE_MIXUP:
                images, targets_a, targets_b, lam = mixup_data(images, labels, ALPHA_MIXUP)

            with autocast(device_type='cuda'):
                outputs = model(images, aux)
                if USE_MIXUP:
                    loss = mixup_criterion(outputs, targets_a, targets_b, lam)
                else:
                    loss = F.cross_entropy(outputs, labels)

                if previous_logits is not None:
                    soft_targets = F.softmax(previous_logits / TEMPERATURE, dim=1)
                    distill_loss = F.kl_div(F.log_softmax(outputs / TEMPERATURE, dim=1), soft_targets, reduction='batchmean')
                    loss += 0.5 * distill_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})

        train_acc = correct / total
        val_acc, _, _, _, _ = evaluate_model(model, test_loader)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"{model_name}_best.pth")

        torch.cuda.empty_cache()
        gc.collect()

    torch.save(model.state_dict(), save_path)
    return model, train_accs, val_accs

# Model Training Wrapper - Updated checkpoint logic
def train_model(model_name, train_csv, test_csv, root_dir, save_path):
    train_ds = SharedHeadDataset(train_csv, root_dir)
    test_ds = SharedHeadDataset(test_csv, root_dir)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fun, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fun, pin_memory=True)

    model = ExtendedModel(model_name).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()

    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_train_acc = 0.0
    previous_logits = None

    for epoch in range(EPOCHS):
        print(f"\n[ Epoch {epoch+1}/{EPOCHS} ]")
        model.train()
        correct, total = 0, 0
        progress = tqdm(train_loader, desc=f"Training {model_name}", leave=False)

        for images, aux, labels in progress:
            images, aux, labels = images.to(DEVICE), aux.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if USE_MIXUP:
                images, targets_a, targets_b, lam = mixup_data(images, labels, ALPHA_MIXUP)

            with autocast(device_type='cuda'):
                outputs = model(images, aux)
                if USE_MIXUP:
                    loss = mixup_criterion(outputs, targets_a, targets_b, lam)
                else:
                    loss = F.cross_entropy(outputs, labels)

                if previous_logits is not None:
                    soft_targets = F.softmax(previous_logits / TEMPERATURE, dim=1)
                    distill_loss = F.kl_div(F.log_softmax(outputs / TEMPERATURE, dim=1), soft_targets, reduction='batchmean')
                    loss += 0.5 * distill_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})

        train_acc = correct / total
        val_acc, _, _, _, _ = evaluate_model(model, test_loader)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save checkpoint if train_acc or val_acc improves
        if (train_acc > best_train_acc) or (val_acc > best_val_acc):
            best_train_acc = max(best_train_acc, train_acc)
            best_val_acc = max(best_val_acc, val_acc)
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"{model_name}_best.pth")

        scheduler.step(epoch)
        torch.cuda.empty_cache()
        gc.collect()

    torch.save(model.state_dict(), save_path)
    return model, train_accs, val_accs 



model_configs = [
    ("swin_small_patch4_window7_224.ms_in1k", "swin_model.pth"),
    ("coatnet_1_rw_224.sw_in1k", "coatnet_model.pth"),
    ("convnext_small.fb_in1k", "convnext_model.pth")
]

from sklearn.model_selection import StratifiedKFold

# Run Stratified K-Fold Training
full_df = pd.read_csv(OUTPUT_DIR / "train_split_augmented.csv")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

FOLDS_DIR = OUTPUT_DIR / "folds"
FOLDS_DIR.mkdir(parents=True, exist_ok=True)

model_id = 2  # Example: ConvNeXt
model_name = model_configs[model_id][0]
history = model_configs[model_id][1]
save_path = ROOT_DIR / "models" / history
plot_save_path = Path.cwd() / "figures" / "shared_head_figures"

# Stratified Fold - Evaluation and Metrics
for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df['label'])):
    print(f"\n Fold {fold + 1} ")
    # Get folds
    train_df = full_df.iloc[train_idx]
    val_df = full_df.iloc[val_idx]

    # Save folds as CSVs
    train_fold_path = FOLDS_DIR / f"{history}_train_fold_{fold+1}.csv"
    val_fold_path = FOLDS_DIR / f"{history}_val_fold_{fold+1}.csv"
    train_df.to_csv(train_fold_path, index=False)
    val_df.to_csv(val_fold_path, index=False)

    # Call training using paths
    model_save_path = save_path / f"{model_name}_fold{fold+1}.pth"
    model_save_path.mkdir(parents=True, exist_ok=True)
    model, train_accs, val_accs = train_model(model_name, train_fold_path, val_fold_path, ROOT_DIR, model_save_path)

    # Plot
    plot_metrics(train_accs, val_accs, model_name, plot_save_path)
    

# Reload and Evaluate
checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pth"
# model = create_model(model_name, pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
model = ExtendedModel(model_name, num_aux_features=3, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(checkpoint_path))

test_loader = DataLoader(
    SharedHeadDataset(TEST_CSV, ROOT_DIR),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fun
)



"""
## ORIGNAL SCRIP (NORMAL EVALUATION)
"""

# # 🧩 Setup: Import and Configuration
# from pathlib import Path
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# from timm import create_model
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import matplotlib.pyplot as plt
# import gc
# import time
# from tqdm import tqdm
# from torch.amp import autocast, GradScaler
# import torch.nn.functional as F
# import random
# import numpy as np

# # ✅ Constants
# IMAGE_SIZE = 224
# BATCH_SIZE = 32
# NUM_CLASSES = 2
# EPOCHS = 15
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# USE_MIXUP = True
# ALPHA_MIXUP = 0.4
# TEMPERATURE = 2.0

# # ✅ Local Paths
# ROOT_DIR = Path("C:/Users/rsriram3/Documents/ind_study")
# OUTPUT_DIR = ROOT_DIR / "data"
# TRAIN_CSV = OUTPUT_DIR / "train_split_augmented.csv"
# TEST_CSV = OUTPUT_DIR / "test_split_augmented.csv"
# CHECKPOINT_DIR = OUTPUT_DIR / "best_checkpoints"
# CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# # 🧺 Dataset Definition (with engineered features)
# class SharedHeadDataset(Dataset):
#     def __init__(self, csv_file, root_dir):
#         self.data = pd.read_csv(csv_file)
#         self.root_dir = Path(root_dir)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         img_path = self.root_dir / row.image
#         try:
#             image = Image.open(img_path)
#         except Exception as e:
#             print(f"[ Failed to load image: {img_path}, error: {e}")
#             return self.__getitem__((idx + 1) % len(self.data))

#         if image.mode != 'RGB':
#             image = image.convert("RGB")

#         transform = transforms.Compose([
#             transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#         image_tensor = transform(image)
#         label = int(row.label)

#         brightness = torch.tensor(row.brightness_mean, dtype=torch.float32).unsqueeze(0)
#         edge_density = torch.tensor(row.edge_density, dtype=torch.float32).unsqueeze(0)
#         entropy = torch.tensor(row.entropy, dtype=torch.float32).unsqueeze(0)

#         aux_features = torch.cat([brightness, edge_density, entropy], dim=0)

#         return image_tensor, aux_features, label

# # 🔄 Collate Function
# def collate_fun(batch):
#     images, aux_feats, labels = zip(*batch)
#     return torch.stack(images), torch.stack(aux_feats), torch.tensor(labels)

# # 🔧 Model Definition
# class ExtendedModel(nn.Module):
#     def __init__(self, backbone_name, num_aux_features=3, num_classes=2):
#         super().__init__()
#         self.backbone = create_model(backbone_name, pretrained=True, num_classes=0)
#         backbone_output_dim = self.backbone.num_features
#         self.backbone.reset_classifier(0)
#         self.classifier = nn.Sequential(
#             nn.Linear(backbone_output_dim + num_aux_features, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x, aux):
#         feats = self.backbone(x)
#         combined = torch.cat([feats, aux], dim=1)
#         return self.classifier(combined)

# # 🧪 Evaluation
# def evaluate_model(model, dataloader):
#     model.eval()
#     preds, targets = [], []
#     with torch.no_grad():
#         for images, aux, labels in dataloader:
#             images, aux = images.to(DEVICE), aux.to(DEVICE)
#             outputs = model(images, aux)
#             preds.extend(torch.argmax(outputs, dim=1).cpu().tolist())
#             targets.extend(labels.tolist())
#     acc = accuracy_score(targets, preds)
#     prec = precision_score(targets, preds)
#     rec = recall_score(targets, preds)
#     f1 = f1_score(targets, preds)
#     cm = confusion_matrix(targets, preds)
#     return acc, prec, rec, f1, cm

# # 📉 Plotting Accuracy
# def plot_metrics(train_accs, val_accs, model_name):
#     plt.figure(figsize=(8, 5))
#     plt.plot(train_accs, label='Train Accuracy')
#     plt.plot(val_accs, label='Validation Accuracy')
#     plt.title(f"Accuracy over Epochs - {model_name}")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"{model_name}_accuracy_plot.png")
#     plt.show()

# # 🔁 Mixup Training
# def mixup_data(x, y, alpha=1.0):
#     lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
#     batch_size = x.size(0)
#     index = torch.randperm(batch_size).to(x.device)
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam

# def mixup_criterion(pred, y_a, y_b, lam):
#     return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)

# # 🏋️ Model Training Wrapper
# def train_model(model_name, train_csv, test_csv, root_dir, save_path):
#     train_ds = SharedHeadDataset(train_csv, root_dir)
#     test_ds = SharedHeadDataset(test_csv, root_dir)
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fun, pin_memory=True)
#     test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fun, pin_memory=True)

#     model = ExtendedModel(model_name).to(DEVICE)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
#     scaler = GradScaler()

#     train_accs, val_accs = [], []
#     best_acc = 0.0
#     previous_logits = None

#     for epoch in range(EPOCHS):
#         print(f"\n[ Epoch {epoch+1}/{EPOCHS} ]")
#         model.train()
#         correct, total = 0, 0
#         progress = tqdm(train_loader, desc=f"Training {model_name}", leave=False)

#         for images, aux, labels in progress:
#             images, aux, labels = images.to(DEVICE), aux.to(DEVICE), labels.to(DEVICE)
#             optimizer.zero_grad()

#             if USE_MIXUP:
#                 images, targets_a, targets_b, lam = mixup_data(images, labels, ALPHA_MIXUP)

#             with autocast(device_type='cuda'):
#                 outputs = model(images, aux)
#                 if USE_MIXUP:
#                     loss = mixup_criterion(outputs, targets_a, targets_b, lam)
#                 else:
#                     loss = F.cross_entropy(outputs, labels)

#                 if previous_logits is not None:
#                     soft_targets = F.softmax(previous_logits / TEMPERATURE, dim=1)
#                     distill_loss = F.kl_div(F.log_softmax(outputs / TEMPERATURE, dim=1), soft_targets, reduction='batchmean')
#                     loss += 0.5 * distill_loss

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             preds = torch.argmax(outputs, dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#             progress.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})

#         train_acc = correct / total
#         val_acc, _, _, _, _ = evaluate_model(model, test_loader)
#         train_accs.append(train_acc)
#         val_accs.append(val_acc)
#         print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

#         scheduler.step(epoch)
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(model.state_dict(), CHECKPOINT_DIR / f"{model_name}_best.pth")

#         torch.cuda.empty_cache()
#         gc.collect()

#     torch.save(model.state_dict(), save_path)
#     return model, train_accs, val_accs

# # ✅ Manual Selection and Training
# model_configs = [
#     ("swin_small_patch4_window7_224.ms_in1k", "swin_model.pth"),
#     ("coatnet_1_rw_224.sw_in1k", "coatnet_model.pth"),
#     ("convnext_small.fb_in1k", "convnext_model.pth")
# ]

# model_id = 2  # Example: ConvNeXt
# model_name = model_configs[model_id][0]
# save_path = ROOT_DIR / "models" / model_configs[model_id][1]

# # 🔁 Train
# model, train_accs, val_accs = train_model(model_name, TRAIN_CSV, TEST_CSV, ROOT_DIR, save_path)

# # 🧪 Reload and Evaluate
# checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pth"
# model = create_model(model_name, pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
# model.load_state_dict(torch.load(checkpoint_path))

# _, _, _, _, cm = evaluate_model(
#     model, DataLoader(SharedHeadDataset(TEST_CSV, ROOT_DIR), batch_size=BATCH_SIZE, collate_fn=collate_fun)
# )
# print("Confusion Matrix:\n", cm)

# # 📉 Plot Metrics
# plot_metrics(train_accs, val_accs, model_name)


