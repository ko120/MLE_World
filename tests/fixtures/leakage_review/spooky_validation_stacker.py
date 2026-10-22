import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from scipy.sparse import hstack
import os
import warnings

warnings.filterwarnings("ignore")

# Create submission directory if it doesn't exist
os.makedirs("./submission", exist_ok=True)

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

print(f"Training data shape: {train.shape}")
print(f"Test data shape: {test.shape}")

# Create stratified train/validation split from training data
train_idx, val_idx = train_test_split(
    train.index, test_size=0.2, stratify=train["author"], random_state=42
)

train_subset = train.loc[train_idx]
val_subset = train.loc[val_idx]

print(f"Train subset shape: {train_subset.shape}")
print(f"Validation subset shape: {val_subset.shape}")

# Character-level TF-IDF vectorization
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 5),
    max_features=6000,
    min_df=2,
    max_df=0.8,
    sublinear_tf=True,
)

# Word-level TF-IDF vectorization
word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=5000,
    min_df=2,
    max_df=0.8,
    sublinear_tf=True,
)

# Character features
X_train_char = char_vectorizer.fit_transform(train_subset["text"])
X_val_char = char_vectorizer.transform(val_subset["text"])
X_test_char = char_vectorizer.transform(test["text"])

# Word features
X_train_word = word_vectorizer.fit_transform(train_subset["text"])
X_val_word = word_vectorizer.transform(val_subset["text"])
X_test_word = word_vectorizer.transform(test["text"])

# Concatenate features
X_train = hstack([X_train_char, X_train_word])
X_val = hstack([X_val_char, X_val_word])
X_test = hstack([X_test_char, X_test_word])

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# Convert labels to integers
label_map = {"EAP": 0, "HPL": 1, "MWS": 2}
y_train = train_subset["author"].map(label_map)
y_val = val_subset["author"].map(label_map)

print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")

# Initialize ensemble models with best hyperparameters from previous solution
models = []

# Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.08,
    min_samples_split=15,
    min_samples_leaf=8,
    subsample=0.85,
    random_state=42,
)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=10,
    random_state=42,
    n_jobs=-1,
)

# Logistic Regression
lr_model = LogisticRegression(
    max_iter=500,
    C=1.0,
    solver="lbfgs",
    random_state=42,
)

models = [gb_model, rf_model, lr_model]

# Fit all models and collect predictions
all_val_probs = []
all_test_probs = []
model_val_losses = []

for i, model in enumerate(models):
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)
    test_pred = model.predict_proba(X_test)
    all_val_probs.append(val_pred)
    all_test_probs.append(test_pred)
    val_loss = log_loss(y_val, val_pred)
    model_val_losses.append(val_loss)
    print(f"Model {i} ({model.__class__.__name__}): validation log loss {val_loss:.6f}")

# Stack all model predictions into proper 2D arrays
all_val_probs_concat = np.hstack(all_val_probs)  # Shape: (3525, 9)
all_test_probs_concat = np.hstack(all_test_probs)  # Shape: (1958, 9)

# Convert predictions to DataFrame for stacking
val_preds_df = pd.DataFrame(
    all_val_probs_concat,
    columns=[f"p_{i}_{j}" for i in range(len(models)) for j in range(3)],
)
test_preds_df = pd.DataFrame(
    all_test_probs_concat,
    columns=[f"p_{i}_{j}" for i in range(len(models)) for j in range(3)],
)

print(f"Meta-features shape (val): {val_preds_df.shape}")
print(f"Meta-features shape (test): {test_preds_df.shape}")

# FIX: Use only the stacked predictions for meta-learner features
# Previously attempted np.vstack of incompatible shapes (9 vs 11000)
val_features = val_preds_df.values
test_features = test_preds_df.values

print(f"Meta-features shape (val): {val_features.shape}")
print(f"Meta-features shape (test): {test_features.shape}")

# Stack a random forest as meta-learner
meta_features = val_features.copy()
y_train_stack = y_train.values
y_val_stack = y_val.values

print("Training meta-learner...")
meta_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)
meta_model.fit(meta_features, y_val_stack)

val_meta_pred = meta_model.predict_proba(meta_features)
test_meta_pred = meta_model.predict_proba(test_features)

# Calculate base model weights based on their individual validation log loss
val_losses = np.array(model_val_losses)
base_weights = 1.0 / (val_losses + 1e-10)
base_weights = base_weights / np.sum(base_weights)
print(f"Base model weights (inverse log loss): {base_weights}")

# Combine meta-model predictions with weighted base model predictions
meta_probs = val_meta_pred
val_probs = (
    meta_probs * 0.7 + np.average(all_val_probs, axis=0, weights=base_weights) * 0.3
)
test_probs = (
    test_meta_pred * 0.7
    + np.average(all_test_probs, axis=0, weights=base_weights) * 0.3
)

print(f"Final val_probs shape: {val_probs.shape}")
print(f"Final test_probs shape: {test_probs.shape}")

# Compute validation multi-class log loss
val_loss = log_loss(y_val, val_probs)
print(f"Validation Multi-class Log Loss: {val_loss:.6f}")

# Generate final submission with proper column order matching sample_submission
submission = pd.DataFrame(
    {
        "id": test["id"],
        "EAP": test_probs[:, 0],
        "HPL": test_probs[:, 1],
        "MWS": test_probs[:, 2],
    }
)

# Rescale probabilities so they sum to 1 for each row
row_sums = submission[["EAP", "HPL", "MWS"]].sum(axis=1)
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = submission[col] / row_sums.values

# Ensure probabilities are in valid range for log loss calculation
epsilon = 1e-15
for col in ["EAP", "HPL", "MWS"]:
    submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

submission = submission.reindex(columns=["id", "EAP", "HPL", "MWS"])

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv with {len(submission)} rows")

print("=== Validation Score ===")
print(f"Validation Log Loss: {val_loss:.6f}")
