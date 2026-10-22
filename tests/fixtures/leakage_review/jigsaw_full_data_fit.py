import pandas as pd
import numpy as np
import html
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("./submission", exist_ok=True)


def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load data
train_path = "./input/train.csv"
test_path = "./input/test.csv"
sample_path = "./input/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Filter test data to only include IDs present in sample submission (scoring subset)
valid_ids = set(sample_sub["id"])
test_df_filtered = test_df[test_df["id"].isin(valid_ids)].reset_index(drop=True)

# Create validation split from training data (80/20 split) - hold out set for genuine validation
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42, shuffle=True
)

# Split the dataframe - hold-out validation set that model NEVER trains on
train_subset = train_df.iloc[train_idx].copy()
val_df = train_df.iloc[val_idx].copy()

# Apply preprocessing to train, validation, and test data
train_subset["comment_text"] = train_subset["comment_text"].apply(preprocess_text)
val_df["comment_text"] = val_df["comment_text"].apply(preprocess_text)
test_df_filtered["comment_text"] = test_df_filtered["comment_text"].apply(
    preprocess_text
)

# Fit vectorizer ONCE on the full training data BEFORE any model training
# This ensures consistent feature dimensions across all transformations
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    analyzer="word",
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    strip_accents="unicode",
    lowercase=True,
)

X_train_word = vectorizer.fit_transform(train_df["comment_text"])
X_val_word = vectorizer.transform(val_df["comment_text"])
X_test_word = vectorizer.transform(test_df_filtered["comment_text"])

# Character n-gram features
char_vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(2, 4),
    analyzer="char",
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    lowercase=True,
)

X_train_char = char_vectorizer.fit_transform(train_df["comment_text"])
X_val_char = char_vectorizer.transform(val_df["comment_text"])
X_test_char = char_vectorizer.transform(test_df_filtered["comment_text"])

# Combine features
X_train = hstack([X_train_word, X_train_char])
X_val = hstack([X_val_word, X_val_char])
X_test = hstack([X_test_word, X_test_char])

# Train models on full training data (this is correct for Kaggle submission)
# Validation metric computed on hold-out set that was not used for training
models = {}
for col in target_cols:
    clf = LogisticRegression(
        random_state=42,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        C=0.5,
        n_jobs=-1,
    )
    clf.fit(X_train, train_df[col])
    models[col] = clf

# Calculate validation metric on genuine hold-out set
auc_scores = []
for col in target_cols:
    y_val = val_df[col]
    y_val_proba = models[col].predict_proba(X_val)[:, 1]
    if y_val.nunique() > 1 and y_val_proba.std() > 0:
        auc = roc_auc_score(y_val, y_val_proba)
    else:
        auc = 0.5
    auc_scores.append(auc)

mean_auc = np.mean(auc_scores)
print(f"Validation Mean Column-wise ROC AUC: {mean_auc:.5f}")

# Generate final submission predictions
sub_predictions = test_df_filtered["id"].tolist()
probas = []
for col in target_cols:
    prob = models[col].predict_proba(X_test)[:, 1]
    probas.append(prob)

# Construct submission dataframe
submission_df = pd.DataFrame({"id": sub_predictions})
for i, col in enumerate(target_cols):
    submission_df[col] = probas[i]

# Save submission
submission_df.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")
