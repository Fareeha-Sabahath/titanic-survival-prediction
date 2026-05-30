# Titanic Survival Prediction Project

## Objective
The objective of this project is to predict passenger survival on the Titanic using machine learning classification models.

---

## Dataset
Titanic Survival Dataset from Kaggle.

Target Variable:
- Survived

---

## Features Used
The dataset contains passenger information such as:
- Age
- Gender
- Passenger Class
- Fare
- Cabin
- Embarked Port
- Family Information

---

## Feature Engineering
The following custom features were created:

### 1. Title Extraction
Passenger titles such as:
- Mr
- Mrs
- Miss
- Master

were extracted from names.

### 2. Family Size
Created using:

```python
FamilySize = SibSp + Parch + 1
```

### 3. Cabin Presence
Created a binary feature indicating whether cabin information exists.

---

## Missing Value Handling
- Missing Age values filled using median
- Missing Embarked values filled using mode
- Cabin information converted into HasCabin feature

---

## Models Used
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier

---

## Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## Project Workflow
1. Data Loading
2. Exploratory Data Analysis
3. Feature Engineering
4. Missing Value Handling
5. Encoding Categorical Variables
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Feature Importance Analysis
10. Model Saving
11. Prediction Example

---

## Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score

---

## Model Explainability
Feature importance analysis was performed using Random Forest feature importances.

---

## Best Model
Random Forest Classifier achieved the best performance.

---

## Model Saving

```python
joblib.dump(rf, 'best_titanic_model.pkl')
```

---

## Example Prediction Code

```python
import pandas as pd
import joblib

model = joblib.load('best_titanic_model.pkl')

sample_passenger = pd.DataFrame(
    [X.iloc[0]],
    columns=X.columns
)

prediction = model.predict(sample_passenger)

print("Survival Prediction:", prediction[0])

if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")
```

---

## Conclusion
This project demonstrates a complete machine learning classification workflow including preprocessing, feature engineering, explainability, model training, evaluation, and prediction using Titanic passenger data.