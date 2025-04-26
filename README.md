# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```

## Output:
DATA HEAD:
![image](https://github.com/user-attachments/assets/41527b75-d96e-4221-8f3e-09eae69b81b7)

DATASET INFO:

![image](https://github.com/user-attachments/assets/da84ce03-be33-4e60-9958-9c18c8066617)

NULL DATASET:

![image](https://github.com/user-attachments/assets/cfe7e3fd-b4c1-4d1b-b742-5a61e6eae760)

VALUES COUNT IN THE LEFT COLUMN:

![image](https://github.com/user-attachments/assets/60fafdc2-df26-4e13-87c7-7320fff5f45b)

DATASET TRANSFORMED HEAD:


![image](https://github.com/user-attachments/assets/a4a32e30-695a-4085-828c-e0de56d4fb49)

X.HEAD:

![image](https://github.com/user-attachments/assets/d0a49ba8-80c8-4db2-9417-3836478b46c8)

ACCURACY:

![image](https://github.com/user-attachments/assets/6d2fab88-15a6-462d-91b8-801bb007822d)

DATA PREDICTION:

![image](https://github.com/user-attachments/assets/7ee74dfe-b24a-43f1-b0e1-32b2ea9d21b1)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
