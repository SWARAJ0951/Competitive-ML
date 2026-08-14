import pandas as pd 

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


