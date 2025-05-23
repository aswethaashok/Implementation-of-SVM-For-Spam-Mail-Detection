# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Start the Program.
Import the necessary packages.
Read the given csv file and display the few contents of the data.
Assign the features for x and y respectively.
Split the x and y sets into train and test sets.
Convert the Alphabetical data to numeric using CountVectorizer.
Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
Find the accuracy of the model.
End the Program.

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: SWETHA A
RegisterNumber:  212223220114

```
```
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:


![Screenshot 2025-05-23 091858](https://github.com/user-attachments/assets/bcce7f69-2917-4b52-a01e-34969d8ef37d)

![Screenshot 2025-05-23 092040](https://github.com/user-attachments/assets/9fe3026b-5bc0-4c1d-8032-81159bdc6e3c)

![Screenshot 2025-05-23 092058](https://github.com/user-attachments/assets/b87fc822-357d-4b59-a116-16c6a8a79977)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
