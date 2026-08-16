import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)


border = "-"*40

##################################
#Step 1 : Load the Dataset
##################################

print(border)
print("Step 1 : Load the Dataset")
print(border)

datapath = "student_performance_ml.csv"
df = pd.read_csv(datapath)

##################################
#Step 2 : Data Analysis EDA
##################################

print(border)
print("Step 2 : Data Analysis EDA")
print(border)

print("Shape of Dataset",df.shape)
print("Columns Name :",list(df.columns))

print("Missing values per columns :")
print(df.isnull().sum())

df = df.drop("SleepHours",axis=1)
print(df.columns)

############################
#Step3 : Decide Independent and Dependent Variables
############################

print(border)
print("Step3 : Decide Independent and Dependent Variables")
print(border)

feature_cols= [
    "StudyHours",
    "Attendance",
    "PreviousScore",
    "AssignmentsCompleted",
]
X = df[feature_cols]
Y= df["FinalResult"]

print("X Shape :",X.shape)
print("Y Shape :",Y.shape)

###########################################
#Step5: Split the DataSet For Training and Testing
##########################################

print(border)
print("Step5: Split the DataSet For Training and Testing")
print(border)

X_test,X_train,Y_test,Y_train = train_test_split(X,Y,test_size=0.2,random_state=42)

print("Dataset spliting activity done ")

###########################################
#Step6: Build The Model
##########################################

print(border)
print("Step6: Build The Model")
print(border)

model  = DecisionTreeClassifier(max_depth=1)
model1 = DecisionTreeClassifier(max_depth=3)
model2 = DecisionTreeClassifier(max_depth=None)

###########################################
#Step7: Train The Model
##########################################

print(border)
print("Step7: Train The Model")
print(border)

model.fit(X_train,Y_train)
model1.fit(X_train,Y_train)
model2.fit(X_train,Y_train)

print(border)

###########################################
#Step8: Test The Model
##########################################

print(border)
print("Step8: Test The Model")
print(border)

Y_pred = model.predict(X_test)
Y_pred1 = model.predict(X_test)
Y_pred2 = model.predict(X_test)

Y_predT = model.predict(X_train)
Y_predT1 = model.predict(X_train)
Y_predT2 = model.predict(X_train)

print("Model Testing Done")
print("Expected answers :")
print(Y_test)

print("Predicted Answer :")
print(Y_pred)

print("Predicted Answer 1 :")
print(Y_pred1)

print("Predicted Answer 2 :")
print(Y_pred2)

###########################################
#Step9: Evaluate The Model Performance (Testing)
##########################################

print(border)
print("Step9: Evaluate The Model Performance Testing")
print(border)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy of model is :",accuracy*100)

print("Confusion Matrix")
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

accuracy1= accuracy_score(Y_test, Y_pred1)
print("Accuracy of model1 is :",accuracy1*100)

print("Confusion Matrix")
cm1 = confusion_matrix(Y_test, Y_pred1)
print(cm1)

accuracy2 = accuracy_score(Y_test, Y_pred2)
print("Accuracy of model2 is :",accuracy2*100)

print("Confusion Matrix")
cm2 = confusion_matrix(Y_test, Y_pred2)
print(cm)

###########################################
#Step10: Evaluate The Model Performance (Training)
##########################################

print(border)
print("Step9: Evaluate The Model Performance Training")
print(border)

accuracyT = accuracy_score(Y_train, Y_predT)
print("AccuracyT of model is :",accuracyT*100)

accuracyT1 = accuracy_score(Y_train, Y_predT1)
print("AccuracyT of model is :",accuracyT1*100)

accuracyT2 = accuracy_score(Y_train, Y_predT2)
print("AccuracyT of model is :",accuracyT2*100)







