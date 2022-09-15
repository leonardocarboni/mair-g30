# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 08:36:44 2022

@author: Merel
"""

import pandas as pd
dialog_act = []
utterance_content= []
with open('dialog_acts(1).dat', 'r') as file:
    for line in file:
        row = line.lower().strip('\n')
        dialog_act.append(row.split(maxsplit = 1)[0])
        utterance_content.append(row.split(maxsplit = 1)[1])
        
d = {'dialog_act': dialog_act, 'utterance_content': utterance_content}
df = pd.DataFrame(data=d)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df['dialog_act'], \
                                                    df['utterance_content'], \
                                                    test_size=0.15, \
                                                    random_state = 42)

"""
Baseline 1:
A baseline system that, regardless of the content of the utterance, 
always assigns the majority class of in the data.
"""
majority = 
while True:
    prompt = input()
    if not prompt:
        break
    else:
        #Assign majority class to the input
        reply = 
        print(reply)
    
"""
Baseline 2:
A baseline rule-based system based on keyword matching.
"""
while True:
    prompt = input()
    if not prompt:
        break
    else:
        #
        majority = 
        reply = 
        print(reply)


