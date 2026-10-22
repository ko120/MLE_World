import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import glob
import PIL.Image
import time

# Constants
NUM_CLASSES = 120
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.0005
LABEL_SMOOTHING = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP_SCALE_INIT = 1.0

# Setup directories
INPUT_DIR = "./input"
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
LABELS_PATH = os.path.join(INPUT_DIR, "labels.csv")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
OUTPUT_SUBMISSION = os.path.join(SUBMISSION_DIR, "submission.csv")

# Load sample submission to get all breed names in correct order
print("Loading submission template...")
sample_sub = pd.read_csv(SAMPLE_SUBMISSION)
breed_names = sample_sub.columns[1:]
breed_to_idx = {breed: idx for idx, breed in enumerate(breed_names)}
idx_to_breed = {idx: breed for breed, idx in breed_to_idx.items()}
NUM_BREEDS = len(breed_names)
print(f"Loaded {NUM_BREEDS} breed classes")

# Load labels
print("Loading labels...")
labels_df = pd.read_csv(LABELS_PATH)
train_ids = set(labels_df["id"].tolist())
print(f"Loaded {len(train_ids)} training images")

# Get test file IDs
test_files = os.listdir(TEST_DIR)
test_ids = set(f.replace(".jpg", "") for f in test_files if f.endswith(".jpg"))
print(f"Found {len(test_ids)} test images")


# Function to load image
def load_image(path, transform):
    img = PIL.Image.open(path).convert("RGB")
    return transform(img)


# Create datasets
class DogDataset(Dataset):
    def __init__(self, image_ids, labels_map, image_dir=None, transform=None):
        self.image_ids = image_ids
        self.labels_map = labels_map
        self.image_dir = image_dir or TRAIN_DIR
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        path = os.path.join(self.image_dir, f"{img_id}.jpg")
        img = load_image(path, self.transform)
        label = self.labels_map.get(img_id)
        if label is None:
            label = -1
        return img, label, img_id


# Initialize breed index mapping for train data
print("Building breed mappings...")
train_breed_idx = {
    row["id"]: breed_to_idx[row["breed"]] for _, row in labels_df.iterrows()
}
print(f"Built mappings for {len(train_breed_idx)} training images")

# Transform functions
train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Split training data - proper 80/20 holdout split
print("Splitting training data...")
train_ids_list = labels_df["id"].tolist()
train_ids_list, val_ids_list = train_test_split(
    train_ids_list, test_size=0.2, random_state=42, shuffle=True
)
print(
    f"Training set: {len(train_ids_list)} images, Validation set: {len(val_ids_list)} images"
)

# Create datasets with correct label mapping
train_dataset = DogDataset(train_ids_list, train_breed_idx, TRAIN_DIR, train_transform)
val_dataset = DogDataset(val_ids_list, train_breed_idx, TRAIN_DIR, val_transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

# Create model using pretrained ResNet50
print("Loading pretrained model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Replace final layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, NUM_BREEDS))

model = model.to(DEVICE)

# Training setup with LABEL SMOOTHING
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("Training model...")
start_time = time.time()

best_val_loss = float("inf")
best_model_state = None

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels, _ in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = correct / total

    # Validation
    model.eval()
    val_logits = []
    val_labels = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            # FIX: Add .detach() before .numpy() to avoid requires_grad error
            val_logits.append(outputs.detach().cpu().numpy())
            val_labels.extend(labels.numpy())

    val_logits = np.vstack(val_logits)
    val_labels = np.array(val_labels)

    # Calculate log loss on validation set (same scale as competition metric)
    val_probs = torch.softmax(torch.tensor(val_logits), dim=1).numpy()
    val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
    val_log_loss = -np.mean(np.log(val_probs[np.arange(len(val_labels)), val_labels]))

    # Save best model
    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        best_model_state = model.state_dict().copy()

    scheduler.step()
    print(
        f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Log Loss: {val_log_loss:.4f}"
    )

# Load best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Loaded best model state from validation")

print(f"Training completed in {time.time() - start_time:.1f} seconds")

# Implement temperature scaling using logistic regression calibration
print("Calibrating model with temperature scaling...")
val_labels_onehot = np.zeros((len(val_labels), NUM_BREEDS), dtype=float)
val_labels_onehot[np.arange(len(val_labels)), val_labels] = 1.0

# Find optimal temperature that minimizes validation log loss
best_temp = TEMP_SCALE_INIT
best_val_loss_temp = float("inf")

for temp in np.logspace(-0.5, 0.5, 20):
    val_logits_scaled = val_logits / temp
    val_probs_calibrated = np.clip(
        torch.softmax(torch.tensor(val_logits_scaled), dim=1).numpy(), 1e-15, 1 - 1e-15
    )
    val_log_loss_temp = -np.mean(
        np.log(val_probs_calibrated[np.arange(len(val_labels)), val_labels])
    )

    if val_log_loss_temp < best_val_loss_temp:
        best_val_loss_temp = val_log_loss_temp
        best_temp = temp

print(f"Optimal temperature: {best_temp:.4f}")

# Create final validation set predictions with temperature scaling for reporting
val_logits = val_logits / best_temp
val_probs = np.clip(
    torch.softmax(torch.tensor(val_logits), dim=1).numpy(), 1e-15, 1 - 1e-15
)
final_log_loss = -np.mean(np.log(val_probs[np.arange(len(val_labels)), val_labels]))

print(f"Training completed in {time.time() - start_time:.1f} seconds")
print(f"Optimal temperature: {best_temp:.4f}")

# Generate final predictions for test set
print("Generating test predictions with temperature scaling...")
test_ids_list = sorted(test_ids)

test_logits = []

for image_id in test_ids_list:
    path = os.path.join(TEST_DIR, f"{image_id}.jpg")
    img = PIL.Image.open(path).convert("RGB")
    img_tensor = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(img)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        test_logits.append(output.detach().cpu().numpy())

test_logits = np.vstack(test_logits)

# Apply temperature scaling to test predictions
test_logits_scaled = test_logits / best_temp
test_probs = np.clip(
    torch.softmax(torch.tensor(test_logits_scaled), dim=1).numpy(), 1e-15, 1 - 1e-15
)

# Create submission file - ensure proper format matching sample_submission.csv
print("Creating submission file...")
submission_rows = []
for img_id, prob_row in zip(test_ids_list, test_probs):
    submission_rows.append([img_id] + [f"{p:.8f}" for p in prob_row])

# Create DataFrame with proper format matching sample_submission.csv columns
submission_data = pd.DataFrame(submission_rows, columns=["id"] + list(breed_names))

# Save submission
submission_data.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"Submission saved to {OUTPUT_SUBMISSION}")

# Print final validation metric - the official Multi-Class Log Loss computed on proper hold-out
final_log_loss_with_temp = -np.mean(
    np.log(np.clip(val_probs[np.arange(len(val_labels)), val_labels], 1e-15, 1 - 1e-15))
)
print(f"\n" + "=" * 50)
print(
    f"HOLD-OUT VALIDATION MULTI-CLASS LOG LOSS (WITH TEMP SCALING): {final_log_loss_with_temp:.4f}"
)
print("=" * 50)
print(f"Validation set size: {len(val_ids_list)} images")
print(f"Test set size: {len(test_ids)} images")
print(f"Optimal temperature: {best_temp:.4f}")
print("=" * 50)
