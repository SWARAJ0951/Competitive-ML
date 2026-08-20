import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


# STEP 1 : LOAD THE DATA
border = "-" * 40


def LoadData(filename):
    df = pd.read_csv(filename)

    print("Data Set Loaded Successfully")
    print(df.head())

    return df


# STEP 2 : CLEAN THE DATASET
def Preprocess(df):

    df.dropna(inplace=True)

    print("Shape of dataset:", df.shape)
    print("Total Records:", df.shape[0])
    print("Total columns:", df.shape[1])

    return df


# STEP 3 : SPLIT THE DATASET FOR TRAINING AND TESTING
def SplitData(df):

    X = df.drop("Class", axis=1)
    Y = df["Class"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    print("Data Splitting Done Successfully")

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("Y_train:", Y_train.shape)
    print("Y_test :", Y_test.shape)

    return X_train, X_test, Y_train, Y_test


# SCALING
# Scaling is required to bring all numerical features
# to the same scale
def scaler(X_train, X_test):

    sc = StandardScaler()

    X_train_scaled = sc.fit_transform(X_train)

    # DO NOT fit scaler again on test data
    X_test_scaled = sc.transform(X_test)

    return X_train_scaled, X_test_scaled


# STEP 4 : HYPER PARAMETER TUNING
def Hyper(X_train_scaled, Y_train, X_test_scaled, Y_test):

    accuracy_scores = []
    K_values = range(1, 21)

    for k in K_values:

        model = KNeighborsClassifier(n_neighbors=k)

        model.fit(X_train_scaled, Y_train)

        Y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(Y_test, Y_pred)

        accuracy_scores.append(accuracy)

    print("Accuracy Report:")

    for k, accuracy in zip(K_values, accuracy_scores):
        print("K =", k, "Accuracy =", accuracy * 100, "%")

    return K_values, accuracy_scores

def main():

    # STEP 1
    df = LoadData("WinePredictor.csv")

    # STEP 2
    df = Preprocess(df)

    # STEP 3
    X_train, X_test, Y_train, Y_test = SplitData(df)

    # STEP 4
    X_train_scaled, X_test_scaled = scaler(
        X_train,
        X_test
    )

    # STEP 5
    K_values, accuracy_scores = Hyper(
        X_train_scaled,
        Y_train,
        X_test_scaled,
        Y_test
    )



if __name__ == "__main__":
    main()