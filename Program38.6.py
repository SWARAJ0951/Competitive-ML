import pandas as pd
import matplotlib.pyplot as plt

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

    #HISOGRAM STUDY HOURS VS FINAL RESULT
    plt.hist(df[df["FinalResult"] == 0]["StudyHours"], alpha=0.5, label="Failed")
    plt.hist(df[df["FinalResult"] == 1]["StudyHours"], alpha=0.5, label="Passed")

    plt.xlabel("Study Hours")
    plt.ylabel("Number of Students")
    plt.title("Study Hours vs Final Result")
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()