import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler

#STEP 1 : LOAD THE DATA 
border = "-"*40

def LoadData(filename):
    df = pd.read_csv(filename)

    print("Data Set Loaded Successfully")

    print(df.head())

    return df

#STEP 2 : CLEAN THE DATASET
def Preprocess(df):

    df.dropna(inplace = True)

    print("Shape of dataset",df.shape)
    print("Total Records :",df.shape[0])
    print("Total columns :",df.shape[1])

    return df

#STEP 3 : SPLIT THE DATA SET FOR TRAINING AND TESTING
def SplitData(df):
    X= df.drop("Class",axis = 1)
    Y= df["Class"]

    X_train , X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    print("Data Spliting Done Succesfully")
    return X_test,X_train,Y_test,Y_train

def main():
   df =  LoadData("WinePredictor.csv")

   df = Preprocess(df)

   X_test,X_train,Y_test,Y_train= SplitData(df)


if __name__ == "__main__":
    main()