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


