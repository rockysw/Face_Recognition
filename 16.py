import pandas as pd
df = pd.read_csv("batch1.csv")
#print(df.loc[:59])
#score_filter = df["Present"]
'''xbox_one_filter =  (df["Present"] == "no")
filtered_reviews = df[xbox_one_filter]
print(filtered_reviews["RollNo"])
  for i in score_filter:
    if(score_filter[i]== "yes"):
        print(score_filter[i])
df['Present'].replace(['yes'],['no'],inplace=True)
print(df["Present"])
df.to_csv('/home/aidl1/FaceID/batch1.csv',index=False)
'''
df.loc[df['RollNo']=="16G01A0505",['Present']]='1'
print(df["Present"])
df.to_csv('/home/aidl1/FaceID/batch1.csv',index=False)
