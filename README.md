# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries.
2. Load the Employee dataset.
3. Preprocess the data (handle categorical variables).
4. Separate features (X) and target variable (y).
5. Split dataset into training and testing sets.
6. Train Decision Tree Classifier.
7. Predict using test data.
8. Evaluate using accuracy and confusion matrix.
   
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KABELAN G K
RegisterNumber: 24900985
*/
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
print("data.info():")
data.info()
print("isnull() and sum():")
data.isnull().sum()
print("data value counts():")
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_ac
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()

```

## Output:
<img width="1683" height="254" alt="Screenshot 2026-02-26 104038" src="https://github.com/user-attachments/assets/3e9a91cd-840d-4b1e-a5a0-d39812402a03" />
<img width="797" height="387" alt="Screenshot 2026-02-26 104105" src="https://github.com/user-attachments/assets/548764ef-b1be-49db-9e70-04601f04c225" />
<img width="688" height="207" alt="Screenshot 2026-02-26 104248" src="https://github.com/user-attachments/assets/ab109ea0-fe10-4c19-9a5f-3ba91fc3cc9f" />
<img width="406" height="91" alt="Screenshot 2026-02-26 104259" src="https://github.com/user-attachments/assets/2643ac25-b947-4c38-b8c4-2162a39c66c4" />
<img width="815" height="146" alt="Screenshot 2026-02-26 104327" src="https://github.com/user-attachments/assets/5ee95aca-6341-4a3d-be7d-47886205624d" />
<img width="363" height="33" alt="Screenshot 2026-02-26 104348" src="https://github.com/user-attachments/assets/9aea052c-2d9f-4f9c-857a-e8106da790a0" />
<img width="558" height="411" alt="Screenshot 2026-02-26 104410" src="https://github.com/user-attachments/assets/51eec748-f489-45b7-b33c-e4cd5e28d070" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
