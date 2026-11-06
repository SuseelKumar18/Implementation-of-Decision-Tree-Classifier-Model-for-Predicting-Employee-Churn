# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm:

1. Start the program and import the required Python libraries.
2. Create a dataset containing employee salary, experience, working hours, and churn status.
3. Split the dataset into training and testing sets.
4. Create and train the Decision Tree Classifier model using the training data.
5. Predict employee churn, evaluate the model using accuracy, and display the results for a new employee.

## Program:
```
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Dataset: Salary, Experience, Working Hours
X = np.array([
    [20000, 1, 10],
    [25000, 2, 9],
    [30000, 3, 8],
    [35000, 4, 8],
    [40000, 5, 7],
    [45000, 6, 7],
    [50000, 7, 6],
    [55000, 8, 6],
    [60000, 9, 5],
    [65000, 10, 5]
])

# Target: Employee Churn (0 = No, 1 = Yes)
y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict employee churn
y_pred = model.predict(X_test)

print("Actual Values:", y_test)
print("Predicted Values:", y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Predict for a new employee
new_employee = np.array([[28000, 2, 9]])
prediction = model.predict(new_employee)

if prediction[0] == 1:
    print("Employee Churn Status: Yes")
else:
    print("Employee Churn Status: No")
```

## Output:
<img width="732" height="367" alt="image" src="https://github.com/user-attachments/assets/88093c29-0482-4133-ae4c-110ed86a1405" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
