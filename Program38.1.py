import pandas as pd

def main():
    datapath = "student_performance_ml.csv"

    df = pd.read_csv(datapath)

    print ("First 5 Records are :",df.head(5))       #first 5 records
    print ("Last 5 Records are :" ,df.tail(5))       #last 5 records
    print ("Total Number of rows and colums are:",df.shape)
    print("Column names :",list(df.columns))
    print (df.dtypes)

if __name__ == "__main__":
    main()