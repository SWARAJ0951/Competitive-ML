import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

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


def main():
   df =  LoadData("WinePredictor.csv")

   df = Preprocess(df)


if __name__ == "__main__":
    main()