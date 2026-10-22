import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X = train.drop(columns=['target'])
y = train['target']
X_T, X_V, y_T, y_V = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_T, y_T)

import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=100)
model.fit(X_T, y_T, eval_set=[(X_V, y_V)], callbacks=[lgb.early_stopping(10)])
validation_predictions = np.clip(y_V.to_numpy(dtype=float), 0.001, 0.999)
print('Held-out validation log loss:', log_loss(y_V, validation_predictions))
