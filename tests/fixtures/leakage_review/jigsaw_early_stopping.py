import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "8"

# Create submission directory
os.makedirs("./submission", exist_ok=True)

# Load data
print("Loading data...")
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

# Split training data into train/val for genuine validation estimate
print("Splitting training data...")
train_idx, val_idx = train_test_split(train.index, test_size=0.2, random_state=42)
train_df = train.loc[train_idx]
val_df = train.loc[val_idx]

# Ensure validation is independent
assert len(train_df) + len(val_df) == len(train), "Split check failed"

# Define target columns
columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

print(
    f"Training set: {len(train_df)}, Validation set: {len(val_df)}, Test set: {len(test)}"
)

# Combined feature pipeline: word unigrams + character 3-grams
print("\nBuilding word TF-IDF features...")
word_vectorizer = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 1), min_df=5, max_df=0.9, sublinear_tf=True
)
X_train_word = word_vectorizer.fit_transform(train_df["comment_text"].astype(str))
X_val_word = word_vectorizer.transform(val_df["comment_text"].astype(str))
X_test_word = word_vectorizer.transform(test["comment_text"].astype(str))
print(f"Word features shape: {X_train_word.shape[1]}")

print("\nBuilding character TF-IDF features (3-grams)...")
char_vectorizer = TfidfVectorizer(
    analyzer="char", ngram_range=(3, 3), max_features=3000, min_df=5, sublinear_tf=True
)
X_train_char = char_vectorizer.fit_transform(train_df["comment_text"].astype(str))
X_val_char = char_vectorizer.transform(val_df["comment_text"].astype(str))
X_test_char = char_vectorizer.transform(test["comment_text"].astype(str))
print(f"Character features shape: {X_train_char.shape[1]}")

print("\nConcatenating features...")
X_train = np.hstack([X_train_word.toarray(), X_train_char.toarray()])
X_val = np.hstack([X_val_word.toarray(), X_val_char.toarray()])
X_test = np.hstack([X_test_word.toarray(), X_test_char.toarray()])
print(
    f"Combined features shape: train={X_train.shape[1]}, val={X_val.shape[1]}, test={X_test.shape[1]}"
)

# Train separate LightGBM for each toxicity type with optimized hyperparameters
print("\nTraining LightGBM models...")
models = {}
for col in columns:
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        num_leaves=20,
        min_child_samples=30,
        random_state=42,
        n_jobs=4,
        verbose=-1,
        class_weight="balanced",
        force_col_wise=True,
    )
    model.fit(
        X_train,
        train_df[col],
        eval_set=[(X_val, val_df[col])],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
    )
    models[col] = model
    print(f"Trained model for {col}")

# Calculate validation ROC AUC on hold-out set (never used for training)
print("\nCalculating validation ROC AUC...")
val_preds = {}
auc_scores = {}
for col in columns:
    preds = models[col].predict_proba(X_val)[:, 1]
    val_preds[col] = preds
    auc = roc_auc_score(val_df[col].values, preds)
    auc_scores[col] = auc
    print(f"{col}: ROC AUC = {auc:.4f}")

mean_auc = np.mean(list(auc_scores.values()))
print(f"\nMean column-wise ROC AUC = {mean_auc:.4f}")

# Generate final submission
print("\nGenerating submission...")
submissions = {}
for col in columns:
    preds = models[col].predict_proba(X_test)[:, 1]
    submissions[col] = preds

submission = pd.DataFrame(
    {"id": test["id"], **{col: subs for col, subs in submissions.items()}}
)
submission = submission.sort_values("id").reset_index(drop=True)

# Ensure directory exists
os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")
print(f"\nFinal submission shape: {submission.shape}")
print(f"Mean validation ROC AUC: {mean_auc:.4f}")
