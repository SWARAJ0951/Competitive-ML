import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

    X = df[['Whether','Temperature']]
    Y = df['Play']

    X_train,X_test,Y_train,Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=10,
        )

    return X_train,X_test,Y_train,Y_test

def TrainModel(X_train,Y_train):
    model = KNeighborsClassifier(n_neighbors=3)
    model = model.fit(X_train,Y_train)
    print("Model Trained Succesfully")

    return model

def TestModel(X_test,model):
    Y_pred = model.predict(X_test)
    print("Model tested Succesfully")

def main():
    df = LoadData("PlayPredictor.csv")

    df = Preprocess(df)

    X_train,X_test,Y_train,Y_test = SplitData(df)

    model = TrainModel(X_train,Y_train)
    model = TrainModel(X_train,Y_train)

    TestModel(X_test,model)

if __name__ == "__main__":
    main()