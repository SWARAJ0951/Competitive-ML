import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
    "SleepHours"
]
X = df[feature_cols]

Y= df["FinalResult"]

print("X Shape :",X.shape)
print("Y Shape :",Y.shape)

###########################################
#Step4: Visualisation Of DataSet
##########################################

print(border)
print("Step4: Visualisation Of DataSet")
print(border)

plt.figure(figsize=(7,5))

plt.scatter(
    df[df["FinalResult"]==1]["StudyHours"],
    df[df["FinalResult"]==1]["Attendance"],
    color = "Green",
    s=100,
    marker="o",
    alpha=0.8,
    edgecolors="black",
    linewidths=1,
    label = "Passed"
    )
plt.scatter(
    df[df["FinalResult"]==0]["StudyHours"],
    df[df["FinalResult"]==0]["Attendance"],
    color = "Red",
    s=100,
    marker="o",
    alpha=0.8,
    edgecolors="black",
    linewidths=1,
    label = "Failed"
    )

plt.xlabel("Study Hours")
plt.ylabel("Attendance")
plt.grid(True)
plt.title("Study Hours vs Attendance")
plt.legend()
plt.show()

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


