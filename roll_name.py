import pandas as pd
df = pd.read_csv("batch1.csv")
temp="16G01A0504"
#df = pd.read_csv("batch1.csv")
xbox_one_filter =  (df["RollNo"] == temp)
filtered_reviews = df[xbox_one_filter]
temp_name=filtered_reviews["Name"]
temp_name = temp_name.split(" ") 
temp_name=temp_name[1]
print(temp_name)
