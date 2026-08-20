import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder

def LoadData(filename):
    df = pd.read_csv(filename)

    print(df.head())
    print("DataSet loaded Successfully")

    return df

def Preprocess(df):
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    le = LabelEncoder()

    df['Whether'] = le.fit_transform(df['Whether'])
    df['Temperature'] = le.fit_transform(df['Temperature'])

    print(df.head())
    return df

def SplitData(df):
    

def TrainModel():
    model = KNeighborsClassifier(n_neighbors=3)
    model = model.fit(X_test)

def main():
    df = LoadData("PlayPredictor.csv")

    df = Preprocess(df)

if __name__ == "__main__":
    main()