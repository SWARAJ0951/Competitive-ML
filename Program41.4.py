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

#scalling is required to bring all numerical features to same scale
def scaler(X_train,X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)


#STEP 4 : TRAIN THE MODEL
def TrainModel(X_train_scaled,Y_train):
    model = KNeighborsClassifier(n_neighbors=9)
    model = model.fit(X_train_scaled,Y_train)

    print("Model Trained Successfully")

    return model

#STEP 5 : EVALUATE THE MODEL
def Evaluate(model,X_test_scaled,Y_test):
    Y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(Y_test,Y_pred)
    print("Accuracy is :",accuracy)


def main():
   df =  LoadData("WinePredictor.csv")

   df = Preprocess(df)

   X_test,X_train,Y_test,Y_train= SplitData(df)

   df = scaler(X_train,X_test)

   model = TrainModel(X_train,Y_train)

   Evaluate(model,X_test,Y_test)



if __name__ == "__main__":
    main()