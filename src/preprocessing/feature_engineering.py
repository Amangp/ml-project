import pandas as pd
import numpy as np

df= pd.read_csv(r"Data\ethereum.csv")

# create new behavorial features

df ["Total Value"]= df["Value_IN(ETH)"]+ df["Value_OUT(ETH)"]
df ["Net Value"]= df["Value_IN(ETH)"]- df["Value_OUT(ETH)"]

epsilon = 1e-9            # we took this so that we caan prevent the division from 0

df["Fee Ratio"] = df["TxnFee(ETH)"] / (df["Total Value"] + epsilon)

df= df.sort_values("UnixTimestamp")
df["Time Gap"]=df["UnixTimestamp"].diff().fillna(0)
df["Block Gap"]=df["Blockno"].diff().fillna(0)

# now we wil standardise z-score

features = ["Total Value","Net Value","Fee Ratio","Time Gap","Block Gap"]
for i in features:
    mean=df[i].mean()
    std=df[i].std()
    df[i+"_z"] = (df[i]-mean)/(epsilon+std)

print("z-score applied")
df.to_csv("Data/processed_ethereum.csv",index= False)
print("New file saved ")