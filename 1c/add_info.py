import random
import pandas as pd

fq = {
    1: "good food",
    2: "bad food",
    3: "mediocre food",
}

cr = {
    1: "busy",
    2: "not busy",
}

ls = {
    1: "long stay",
    2: "short stay"
}

df = pd.read_csv("1c/restaurant_info.csv")

df.insert(df.shape[1], "food_quality", [fq[random.randint(1, 3)] for i in range(len(df))])
df.insert(df.shape[1], "crowdedness", [cr[random.randint(1, 2)] for i in range(len(df))])
df.insert(df.shape[1], "stay_length", [ls[random.randint(1, 2)] for i in range(len(df))])

df.to_csv("1c/restaurant_info2.csv", index=False)