import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('heart.csv')

# Separate features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Initialize models
log_reg = LogisticRegression(max_iter=1000)
nb = GaussianNB()

# Apply k-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Logistic Regression
log_reg_pred = cross_val_predict(log_reg, X, y, cv=kf, method='predict')
log_reg_prob = cross_val_predict(log_reg, X, y, cv=kf, method='predict_proba')[:, 1]

# Naive Bayes
nb_pred = cross_val_predict(nb, X, y, cv=kf, method='predict')
nb_prob = cross_val_predict(nb, X, y, cv=kf, method='predict_proba')[:, 1]

# Confusion Matrices
log_reg_cm = confusion_matrix(y, log_reg_pred)
nb_cm = confusion_matrix(y, nb_pred)

# Classification Reports
log_reg_report = classification_report(y, log_reg_pred)
nb_report = classification_report(y, nb_pred)

# Print confusion matrices and classification reports
print("Logistic Regression Confusion Matrix:")
print(log_reg_cm)
print("\nLogistic Regression Classification Report:")
print(log_reg_report)

print("Naive Bayes Confusion Matrix:")
print(nb_cm)
print("\nNaive Bayes Classification Report:")
print(nb_report)

# Plot ROC and AUC curves
log_reg_fpr, log_reg_tpr, _ = roc_curve(y, log_reg_prob)
nb_fpr, nb_tpr, _ = roc_curve(y, nb_prob)

log_reg_auc = auc(log_reg_fpr, log_reg_tpr)
nb_auc = auc(nb_fpr, nb_tpr)

plt.figure(figsize=(10, 6))
plt.plot(log_reg_fpr, log_reg_tpr, label=f'Logistic Regression (AUC = {log_reg_auc:.2f})')
plt.plot(nb_fpr, nb_tpr, label=f'Naive Bayes (AUC = {nb_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()