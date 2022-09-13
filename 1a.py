import pandas as pd
from sklearn.model_selection import train_test_split

# setup

dialog_act = []
utterance_content= []
with open('dialog_acts.dat', 'r') as file:
    for line in file:
        row = line.lower().strip('\n')
        dialog_act.append(row.split(maxsplit = 1)[0])
        utterance_content.append(row.split(maxsplit = 1)[1])

d = {'dialog_act': dialog_act, 'utterance_content': utterance_content}

df = pd.DataFrame(data=d)

# dataframe
X_train, X_test, Y_train, Y_test = train_test_split(df['dialog_act'], df['utterance_content'], test_size=0.15, random_state = 42)

