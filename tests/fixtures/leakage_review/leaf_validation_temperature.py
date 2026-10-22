import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import xgboost as xgb
from scipy.optimize import minimize_scalar

# Create submission directory if it doesn't exist
os.makedirs("./submission", exist_ok=True)

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

# Get feature columns (all columns except id and species)
feature_cols = [col for col in train_df.columns if col not in ["id", "species"]]

# Extract features and labels
X_train = train_df[feature_cols]
y_train = train_df["species"]
X_test = test_df[feature_cols]
test_ids = test_df["id"]

# Get all class labels from sample submission for consistent ordering
class_labels = sample_submission.columns[1:]  # Skip 'id' column
n_classes = len(class_labels)
print(f"Number of classes: {n_classes}")

# Create mapping from label names to indices
label_to_idx = {label: idx for idx, label in enumerate(class_labels)}

# Encode training labels to indices
y_train_encoded = y_train.map(label_to_idx).values

# Hold-out validation split - model trains ONLY on training portion
X_train_val, X_val, y_train_val, y_val = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

print(f"Training set size: {X_train_val.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# Train XGBoost model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model.fit(X_train_val, y_train_val)

# Make predictions on validation set
val_pred_proba = model.predict_proba(X_val)
test_pred_proba = model.predict_proba(X_test)

print(f"Validation prediction shape: {val_pred_proba.shape}")
print(f"Test prediction shape: {test_pred_proba.shape}")


# Temperature scaling calibration
def temperature_loss_fn(val_proba, y_true_onehot, temperature):
    """Compute log-loss with temperature scaling applied."""
    scaled_proba = val_proba ** (1.0 / temperature)
    scaled_proba = np.clip(scaled_proba, 1e-15, 1 - 1e-15)
    scaled_proba = scaled_proba / scaled_proba.sum(axis=1, keepdims=True)
    loss = -np.mean(np.sum(y_true_onehot * np.log(scaled_proba), axis=1))
    return loss


# Create one-hot encoded validation labels
y_val_onehot = np.zeros((len(y_val), n_classes))
for i, label_idx in enumerate(y_val):
    y_val_onehot[i, label_idx] = 1.0


# Optimize temperature parameter
def objective(temperature):
    return temperature_loss_fn(val_pred_proba, y_val_onehot, temperature)


result = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
optimal_temperature = result.x
print(f"Optimal temperature: {optimal_temperature:.4f}")

# Apply temperature scaling to test predictions
test_pred_proba_scaled = test_pred_proba ** (1.0 / optimal_temperature)
# Clip to avoid log(0) issues
test_pred_proba_scaled = np.clip(test_pred_proba_scaled, 1e-15, 1 - 1e-15)
# Normalize each row to sum to 1 (CRITICAL FIX - this was missing before)
test_pred_proba_calibrated = test_pred_proba_scaled / test_pred_proba_scaled.sum(
    axis=1, keepdims=True
)

# Apply temperature scaling to validation predictions
val_pred_proba_scaled = val_pred_proba ** (1.0 / optimal_temperature)
val_pred_proba_scaled = np.clip(val_pred_proba_scaled, 1e-15, 1 - 1e-15)
val_pred_proba_calibrated = val_pred_proba_scaled / val_pred_proba_scaled.sum(
    axis=1, keepdims=True
)

# Create submission file with proper format
submission = pd.DataFrame()
submission["id"] = test_ids.values

# Ensure prediction columns match sample submission format (same order)
for class_name in class_labels:
    col_idx = label_to_idx[class_name]
    submission[class_name] = test_pred_proba_calibrated[:, col_idx]

# Save to submission directory
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv with shape {submission.shape}")
print(
    f"Columns: {list(submission.columns[:5])}... (total {len(submission.columns)} columns)"
)

# Verify row sums
row_sums = submission.iloc[:, 1:].sum(axis=1)
print(
    f"Submission row sums - min: {row_sums.min():.6f}, max: {row_sums.max():.6f}, mean: {row_sums.mean():.6f}"
)


# Compute validation log-loss for reporting (on hold-out set, not resubstituted)
def compute_log_loss(predictions, true_labels_onehot):
    """
    Compute multi-class logarithmic loss as defined in competition:
    logloss = -1/N * sum(sum(y_ij * log(p_ij)))
    """
    N = len(predictions)

    # Clamp predictions to avoid log(0) issues
    predictions_clamped = np.clip(predictions, 1e-15, 1 - 1e-15)

    log_loss_val = -np.mean(
        np.sum(true_labels_onehot * np.log(predictions_clamped), axis=1)
    )
    return log_loss_val


validation_logloss = compute_log_loss(val_pred_proba_calibrated, y_val_onehot)
print(
    f"Validation Multi-class Log-Loss (calibrated on hold-out): {validation_logloss:.6f}"
)
