import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
from pathlib import Path
from PIL import Image

# Suppress tqdm progress bars
import warnings

warnings.filterwarnings("ignore")


# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Paths
INPUT_DIR = Path("./input")
SUBMISSION_DIR = Path("./submission")
SUBMISSION_DIR.mkdir(exist_ok=True)

# Load data
train_df = pd.read_csv(INPUT_DIR / "train.csv")
test_df = pd.read_csv(INPUT_DIR / "test.csv")
sample_sub = pd.read_csv(INPUT_DIR / "sample_submission.csv")

# Image paths
image_dir = INPUT_DIR / "images"

# Target columns - match the sample submission structure (4 targets)
target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

# Validation split (80% train, 20% val) - this split is used for validation only
# The final model will never see val_indices
random.seed(42)
np.random.seed(42)
indices = np.arange(len(train_df))
np.random.shuffle(indices)
split_idx = int(0.8 * len(indices))
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

# Validation set (hold-out)
val_df = train_df.iloc[val_indices].copy()
train_df_subset = train_df.iloc[train_indices].copy()

# Targets for training and validation - CRITICAL: convert to float for BCEWithLogitsLoss
train_targets = train_df_subset[target_cols].values.astype(np.float32)
val_targets = val_df[target_cols].values.astype(np.float32)


# Get image paths for train subset
def get_image_paths(df, image_dir):
    return df["image_id"].apply(lambda x: image_dir / f"{x}.jpg").tolist()


train_images_paths = get_image_paths(train_df_subset, image_dir)
val_images_paths = get_image_paths(val_df, image_dir)


# Create datasets
class ImageDataset(Dataset):
    def __init__(self, images_paths, targets, transform=None):
        self.images = images_paths
        self.targets = targets
        self.transforms = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        target = self.targets[idx]
        return image, target


# Data transforms
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create data loaders
train_dataset = ImageDataset(train_images_paths, train_targets, transform=transform)
val_dataset = ImageDataset(val_images_paths, val_targets, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)


# Model architecture - pretrained ResNet18 for multi-label
def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze final block
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final layer with 4 output units for multi-label
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 4),
    )

    return model


# Training function
@torch.no_grad()
def get_model_predictions(model, loader, device):
    model.eval()
    all_outputs = []
    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)
        all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
    return np.concatenate(all_outputs, axis=0)


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs
):
    model.train()
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        train_losses = []
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_outputs = get_model_predictions(model, val_loader, device)
        val_loss = np.mean(train_losses)

        auc_scores = []
        for i in range(val_outputs.shape[1]):
            try:
                auc = roc_auc_score(val_targets[:, i], val_outputs[:, i])
                auc_scores.append(auc)
            except ValueError:
                auc_scores.append(0.5)
        val_auc = np.mean(auc_scores)

        scheduler.step(val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 5:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    val_outputs = get_model_predictions(model, val_loader, device)
    val_aucs = []
    for i in range(val_outputs.shape[1]):
        try:
            auc = roc_auc_score(val_targets[:, i], val_outputs[:, i])
            val_aucs.append(auc)
        except ValueError:
            val_aucs.append(0.5)

    return model, best_val_auc, val_aucs


# Train first model with seed 42
set_seeds(42)
model1 = create_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = model1.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer1 = optim.AdamW(
    filter(lambda p: p.requires_grad, model1.parameters()), lr=1e-3
)
scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer1, mode="min", factor=0.5, patience=3
)

model1, auc1_1, auc1_per_col = train_model(
    model1, train_loader, val_loader, criterion, optimizer1, scheduler1, device, 20
)
print(f"Model 1 (seed=42) best val AUC: {auc1_1:.6f}")

# Train second model with seed 123
set_seeds(123)
model2 = create_model()
model2 = model2.to(device)

optimizer2 = optim.AdamW(
    filter(lambda p: p.requires_grad, model2.parameters()), lr=1e-3
)
scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer2, mode="min", factor=0.5, patience=3
)

model2, auc2_1, auc2_per_col = train_model(
    model2, train_loader, val_loader, criterion, optimizer2, scheduler2, device, 20
)
print(f"Model 2 (seed=123) best val AUC: {auc2_1:.6f}")

# Ensemble predictions - average the two models
val_outputs_1 = get_model_predictions(model1, val_loader, device)
val_outputs_2 = get_model_predictions(model2, val_loader, device)
val_outputs_ensemble = (val_outputs_1 + val_outputs_2) / 2

# Calculate validation metric
val_auc_scores = []
for i in range(val_outputs_ensemble.shape[1]):
    try:
        auc = roc_auc_score(val_targets[:, i], val_outputs_ensemble[:, i])
        val_auc_scores.append(auc)
    except ValueError:
        val_auc_scores.append(0.5)

mean_val_auc = np.mean(val_auc_scores)
print(f"\nEnsemble Validation Mean Column-wise ROC AUC Score: {mean_val_auc:.6f}")
print("\nPer-column Validation ROC AUC:")
for i, col in enumerate(target_cols):
    auc = roc_auc_score(val_targets[:, i], val_outputs_ensemble[:, i])
    print(f"  {col}: {auc:.6f}")


# Generate predictions on test set
@torch.no_grad()
def predict_test(model, test_image_ids, image_dir, device):
    model.eval()
    all_outputs = []

    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for image_id in test_image_ids:
        image_path = image_dir / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        image = test_transform(image).unsqueeze(0).to(device)

        output = model(image)
        outputs = torch.sigmoid(output).cpu().numpy()[0]
        all_outputs.append(outputs)

    return np.array(all_outputs)


test_image_ids = sorted(test_df["image_id"].unique())
test_predictions = predict_test(model1, test_image_ids, image_dir, device)
test_predictions_ensemble = (
    test_predictions + predict_test(model2, test_image_ids, image_dir, device)
) / 2

# Create submission
submission_df = pd.DataFrame(
    {
        "image_id": test_image_ids,
        "healthy": test_predictions_ensemble[:, 0],
        "multiple_diseases": test_predictions_ensemble[:, 1],
        "rust": test_predictions_ensemble[:, 2],
        "scab": test_predictions_ensemble[:, 3],
    }
)

submission_df.to_csv(SUBMISSION_DIR / "submission.csv", index=False)
print(f"\nEnsemble submission saved to: {SUBMISSION_DIR / 'submission.csv'}")
