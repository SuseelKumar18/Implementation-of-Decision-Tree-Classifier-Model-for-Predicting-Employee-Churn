# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1) Import pandas module and import the required data set.
2) Find the null values and count them.
3) Count number of left values.
4) From sklearn import LabelEncoder to convert string values to numerical values.
5) From sklearn.model_selection import train_test_split.
6) Assign the train dataset and test dataset.
7) From sklearn.tree import DecisionTreeClassifier.
8) Use criteria as entropy.
9) From sklearn import metrics.
10) Find the accuracy of our model and predict the require values.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: R . Nithish Aaditiyaa
RegisterNumber:  25011876[212225040287]
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
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()


y=data["left"]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy value:", accuracy)


print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()


```

## Output:
<img width="722" height="605" alt="image" src="https://github.com/user-attachments/assets/55240a38-badb-4077-af95-fc0115f26227" />


<img width="818" height="605" alt="image" src="https://github.com/user-attachments/assets/40d44b35-8095-4124-bb05-0603ba39b3e6" />


## Result:

Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

