# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 08:36:44 2022

@author: Merel
"""

import pandas as pd
dialog_act = []
utterance_content= []
with open('dialog_acts.dat', 'r') as file:
    for line in file:
        row = line.lower().strip('\n')
        dialog_act.append(row.split(maxsplit = 1)[0])
        utterance_content.append(row.split(maxsplit = 1)[1])
        
d = {'dialog_act': dialog_act, 'utterance_content': utterance_content}
df = pd.DataFrame(data=d)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df['utterance_content'], \
                                                    df['dialog_act'], \
                                                    test_size=0.15, \
                                                    random_state = 42)

"""
Baseline 1:
A baseline system that, regardless of the content of the utterance, 
always assigns the majority class of in the data.
"""
majority = Y_train.mode()[0]
while True:
    prompt = input()
    if not prompt:
        break
    else:
        print(majority)
    
"""
Baseline 2:
A baseline rule-based system based on keyword matching.
"""

classes = {
    'ack': ['ok', 'okay'],
    'affirm': ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay'],
    'bye': ['bye', 'goodbye', 'see you', 'see you later', 'talk to you later'],
    'confirm': [],
    'deny': ['dont', 'don\'t', 'do not'],
    'hello': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
    'inform': ['look', 'looking for'],
    'negate': ['no'],
    'null': ['cough', 'clear throat', 'laugh', 'sigh', 'sniff'],
    'repeat': ['again', 'repeat', 'what was that', 'what did you say', 'what did you mean'],
    'reqalts': ['alternatives', 'other options', 'other choices', 'other suggestions'],
    'reqmore': ['more', 'another', 'different', 'else', 'other'],
    'request': ["?"],
    'restart': ['start over', 'restart', 'start again', 'start from the beginning'],
    'thankyou': ['thank you', 'thanks', 'thank you very much', 'thanks very much'],
}
while True:
    prompt = input()
    if not prompt:
        break
    else:
        for key, value in classes.items():
            for v in value:
                if v in prompt:
                    print(key)
                    break
            else:
                continue
            break
        else:
            print(majority)


