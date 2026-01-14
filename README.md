# Rock vs Mine Prediction using Machine Learning

## Overview

This project uses **Logistic Regression**, a supervised learning algorithm, to classify sonar signals as either **Rock (R)** or **Mine (M)** based on frequency data.  
The dataset consists of 60 numerical features representing sonar echo intensities at various angles.  
The goal is to train a binary classifier that distinguishes between metal (mines) and rock objects beneath the sea.

---

## Dataset Information

**File:** `sonar_data.csv`  
**Source:** UCI Machine Learning Repository  
**Details:**

- 208 samples
- 60 continuous features (sonar frequency readings)
- 1 label column (`R` = Rock, `M` = Mine)

Each instance is a set of sonar readings bounced off a surface — the reflections differ depending on whether the surface is metallic (mine) or rocky.

---

## Project Structure

```

Rock_vs_Mine_Prediction/
│
├── Rock_vs_Mine_Prediction.ipynb   # Main Jupyter Notebook
├── sonar_data.csv                  # Dataset
├── README.md                       # Project documentation
└── requirements.txt                # Dependencies (optional)

```

---

## Implementation Steps

### 1. Import Libraries

Essential libraries for data processing, model building, and evaluation.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### 2. Load Dataset

```python
data = pd.read_csv("sonar_data.csv", header=None)
```

### 3. Data Inspection

Check shape, head, and basic statistics.

```python
print(data.shape)
print(data.head())
```

### 4. Split Features and Labels

```python
X = data.drop(columns=60, axis=1)
y = data[60]
```

### 5. Label Encoding

Convert categorical labels (`R`, `M`) to numeric form.

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
```

### 6. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=1
)
```

### 7. Model Training

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 8. Model Evaluation

```python
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print("Training Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
```

### 9. Predicting a New Sample

```python
input_data = (...60 feature values...)
input_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_array)
print("Mine" if prediction[0] == 1 else "Rock")
```

---

## Example Output

```
Training Accuracy: 0.83
Test Accuracy: 0.76
Predicted object: Mine
```

---

## Key Concepts

**Logistic Regression:**

- A linear model for binary classification.
- Outputs probabilities using a **sigmoid function**.
- Decision boundary determined by a probability threshold (default 0.5).

**Data Splitting:**

- Ensures fair model evaluation.
- `stratify=y` maintains class ratio in training and testing subsets.

**Accuracy Score:**

- Measures the percentage of correct predictions.

---

## Dependencies

List of Python packages used:

```
numpy
pandas
scikit-learn
```

Install them with:

```bash
pip install numpy pandas scikit-learn
```

---

## Usage

### Run Notebook

```bash
jupyter notebook Rock_vs_Mine_Prediction.ipynb
```

### Or Run Script

Save the `.py` version of the notebook and execute:

```bash
python Rock_vs_Mine_Prediction.py
```

---

## Results Summary

- Logistic Regression provides interpretable, quick results for binary sonar classification.
- Model performs well on small datasets with linear separability.
- Accuracy can be further improved with:
  - Feature scaling
  - Hyperparameter tuning
  - Advanced models (SVM, Random Forest, Neural Networks)
