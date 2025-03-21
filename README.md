# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: A Ahil Santo

RegisterNumber: 212224040018

```
import pandas as pd
data=pd.read_csv(r"E:\Desktop\CSE\Introduction To Machine Learning\dataset\Placement_Data.csv")
print("Placement DataSet:\n",data.head())
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
print("Dataset after dropping sl_no and salary: \n",data1.head())
print("Check for Missing and Duplicate Values:")
print(data1.isnull())
print("\nNo of Duplicate entries: ",data1.duplicated().sum())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"]) 
data1["hsc_b"]=le.fit_transform(data1["hsc_b"]) 
data1["hsc_s"]=le.fit_transform(data1["hsc_s"]) 
data1["degree_t"]=le.fit_transform(data1["degree_t"]) 
data1["workex"]=le.fit_transform(data1["workex"]) 
data1["specialisation"]=le.fit_transform(data1["specialisation"]) 
data1["status"]=le.fit_transform(data1["status"])

x=data1.iloc[:,:-1]
y=data1["status"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Y predicition: ",y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy of the model: {accuracy}")

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

```

## Output:
Placement Dataset:

![1](https://github.com/user-attachments/assets/23821d7b-7ded-4446-87ed-ac5332c6a4bb)

![2](https://github.com/user-attachments/assets/486b463f-0500-47f3-83d6-ad5c87670a0a)

Missing and Duplication of Data:

![3](https://github.com/user-attachments/assets/903e2b17-8eb9-4cf2-b86a-0460c5be162f)

Y Prediction Value:

![4](https://github.com/user-attachments/assets/c9bf1cb4-d3ee-477d-9670-6e15185be973)

Accuracy and Classification_report:

![5](https://github.com/user-attachments/assets/c5819b23-78ff-48d5-8637-cd9305ebf4da)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
