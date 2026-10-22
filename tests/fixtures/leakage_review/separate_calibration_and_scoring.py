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

X_fit, X_cal, y_fit, y_cal = train_test_split(X_T, y_T, test_size=0.2, random_state=43)
model.fit(X_fit, y_fit)
cal_logits = model.decision_function(X_cal)
def calibrated_probability(logits, temperature):
    return 1 / (1 + np.exp(-logits / temperature))
temperature = minimize_scalar(lambda t: log_loss(y_cal, calibrated_probability(cal_logits, t)), bounds=(0.1, 5), method='bounded').x
validation_loss = log_loss(y_V, calibrated_probability(model.decision_function(X_V), temperature))
print('Held-out validation log loss:', validation_loss)
