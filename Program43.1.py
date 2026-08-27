import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder

def LoadData(filename):
    df = pd.read_csv(filename)

    print(df.head())
    print("DataSet loaded Successfully")

def main():
    LoadData("PlayPredictor.csv")

if __name__ == "__main__":
    main()