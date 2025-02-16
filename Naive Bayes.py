import pandas as pd
import numpy as np

df = pd.read_csv('census-income-dataset.csv')

x = df.iloc[:, :14]  # Features
y = df.iloc[:, 14:]  # Labels

class_probs = {}
feature_probs = {}

# Count the occurrences of each class and feature
for i, row in enumerate(x.values):
    label = y.iloc[i, 0]  # Get label
    if label not in class_probs:
        class_probs[label] = 0
        feature_probs[label] = {}
    class_probs[label] += 1

    for j, value in enumerate(row):
        if j not in feature_probs[label]:
            feature_probs[label][j] = {}
        if value not in feature_probs[label][j]:
            feature_probs[label][j][value] = 0
        feature_probs[label][j][value] += 1

# Calculate probabilities
for label in class_probs:
    total_samples = class_probs[label]
    for feature in feature_probs[label]:
        for value in feature_probs[label][feature]:
            feature_probs[label][feature][value] /= total_samples

# Test samples
samples = [
    [39, 'State-gov', 77516, 'Bachelors', 13, 'Never-married', 'Adm-clerical', 'Not-in-family', 'White', 'Male', 2174, 0, 40, 'United-States'],
    [31, 'Private', 45781, 'Masters', 14, 'Never-married', 'Prof-speciality', 'Not-in-family', 'White', 'Female', 14084, 0, 50, 'United-States'],
    [42, 'Private', 15949, 'Bachelors', 13, 'Married-civ-spouse', 'Exec-managerial', 'Husband', 'White', 'Male', 5178, 0, 40, 'United-States']
]

# Convert samples to numerical format
X_test_numerical = []
for sample in samples:
    numerical_row = []
    for i, value in enumerate(sample):
        if isinstance(value, str):
            numerical_row.append(i)  # For string features, use the index as a placeholder
        else:
            numerical_row.append(value)
    X_test_numerical.append(numerical_row)

# Make predictions
for i, sample in enumerate(X_test_numerical):
    best_label = None
    best_score = float('-inf')
    for label in class_probs:
        score = np.log(class_probs[label])
        for j, value in enumerate(sample):
            if isinstance(value, int):
                score += np.log(feature_probs[label][j][value] if value in feature_probs[label][j] else 1e-10)
        if score > best_score:
            best_label = label
            best_score = score
            predicted_prob = np.exp(best_score)
    print(f"For sample {i+1} - {samples[i]}:")
    print(f"Predicted label: {best_label}\n Score: {predicted_prob}, {np.exp(score)}")
