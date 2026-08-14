import pandas as pd
import matplotlib.pyplot as plt

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


