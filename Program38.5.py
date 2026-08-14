import pandas as pd

def main():
    datapath = "student_performance_ml.csv"

    df = pd.read_csv(datapath)

    print ("First 5 Records are :",df.head(5))       #first 5 records
    print ("Last 5 Records are :" ,df.tail(5))       #last 5 records
    print ("Total Number of rows and colums are:",df.shape)
    print("Column names :",list(df.columns))
    print (df.dtypes)
    print("Total Number of Students are :",df.shape[0])

    matching_rows = (df["FinalResult"]== 1).sum()
    print("Studnets Passed :",matching_rows)

    matching_rows = (df["FinalResult"]== 0).sum()
    print("Studnets Failed :",matching_rows)

    avearge = df["StudyHours"].mean()
    print("Average Study Hours ",avearge) 

    avearge = df["Attendance"].mean()
    print("Average Attendance ",avearge)  

    maximum = df["PreviousScore"].max()
    print("Maximum Previous Score :",maximum)

    minimum = df["SleepHours"].min()
    print("Minimum SleepHours :",minimum)


    #NO THE DATASET IS NOT BALANCE BECAUSE PASSED STUDENTS ARE 60% AND FAILED ARE 40 %
    value = df["FinalResult"].value_counts(normalize=True)*100
    print(value)


    #groupby divide the data and finds the resut
    std = df.groupby("StudyHours")["FinalResult"].mean()
    print(std)

    std = df.groupby("Attendance")["FinalResult"].mean()
    print(std)

    

if __name__ == "__main__":
    main()