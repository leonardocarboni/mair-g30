# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:15:01 2022

@author: bais_
"""


"""
Part 1a: text classification
Group G30 (Giacomo Bais, Leonardo Carboni, Merel de Goede, Merel van den Bos)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Creating the dataframe
d = pd.read_csv('dialog_acts.dat', header=None)
df = pd.DataFrame(data=d)
df.columns = ['dialog_act']

# Splitting the dataframe columns
df[['dialog_act', 'utterance_content']
   ] = df.dialog_act.str.split(' ', 1, expand=True)

# Lowercasing the content
df['dialog_act'] = df['dialog_act'].str.lower()
df['utterance_content'] = df['utterance_content'].str.lower()

# Splitting the dataframe into train and test
X_train, X_test, Y_train, Y_test = train_test_split(df['utterance_content'],
                                                    df['dialog_act'],
                                                    test_size=0.15,
                                                    random_state=42)

# Majority class
majority = Y_train.mode()[0]

# Classes Dictionary
classes = {
    'ack': ['ok', 'okay', "fine"],
    'affirm': ['yes', 'yeah', 'yep', 'sure', 'right', 'indeed'],
    'bye': ['bye', 'goodbye', 'see', 'talk'],
    'confirm': ['true'],
    'deny': [ 'don\'t','cannot', 'cant', 'can\'t', 'no', 'nope', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor', 'never', 'none', 'hardly', 'scarcely', 'barely', 'rarely', 'seldom', 'little', 'less', 'least', 'few', 'fewer', 'fewest', 'short', 'shorter', 'shortest', 'low', 'lower', 'lowest', 'small', 'smaller', 'smallest', 'tiny', 'tiniest', 'young', 'younger', 'youngest', 'little', 'less', 'least', 'few', 'fewer', 'fewest', 'short', 'shorter', 'shortest', 'low', 'lower', 'lowest', 'small', 'smaller', 'smallest', 'tiny', 'tiniest', 'young', 'younger', 'youngest', 'little', 'less', 'least', 'few', 'fewer', 'fewest', 'short', 'shorter', 'shortest', 'low', 'lower', 'lowest', 'small', 'smaller', 'smallest', 'tiny', 'tiniest', 'young', 'younger', 'youngest', 'little', 'less', 'least', 'few', 'fewer', 'fewest', 'short', 'shorter', 'shortest', 'low', 'lower', 'lowest', 'small', 'smaller', 'smallest', 'tiny', 'tiniest', 'young', 'younger', 'youngest', 'little', 'less', 'least', 'few', 'fewer', 'fewest', 'short', 'shorter', 'shortest', 'low', 'lower', 'lowest', 'small', 'smaller', 'smallest', 'tiny', ],
    'hello': ['hello', 'hi', 'hey', 'morning', 'afternoon', 'evening'],
    'inform': ['look', 'looking', 'search', 'find', 'want', 'need', 'require', 'requirement', 'requirements'],
    'negate': ['no', 'nope', 'not', 'never', 'none', 'nothing'],
    'null': ['cough', 'clear', 'laugh', 'sigh', 'sniff', 'noise', 'sil', 'unintelligible'],
    'repeat': ['again', 'repeat'],
    'reqalts': ['about', 'alternatives', 'other','another', 'different', 'else', 'other'],
    'reqmore': ['more', 'else', 'another', 'different', 'anything'],
    'request': ["whats","?", "what", "train", "taxi", "plane", 'phone'],
    'restart': ['start', 'restart', 'again', 'beginning'],
    'thankyou': ['thank', 'thanks', 'thankyou'],
}


# Main Loop
while True:

    # User Input
    print("\nChoose what you want to do:")
    print("1. Baseline 1")
    print("2. Baseline 2")
    print("3. Test")
    print("\n0. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        """
        Baseline 1:
        A baseline system that, regardless of the content of the utterance, 
        always assigns the majority class of in the data.
        """
        print("\nBaseline 1 - Write a sentence or a word")

        # Reading input and converting it in lower case
        prompt = input().lower()
        print(majority)

    elif choice == "2":
        """
        Baseline 2:
        A baseline rule-based system based on keyword matching.
        """
        print("\nBaseline 2 - Write a sentence or a word")

        # Reading input and converting it in lower case
        prompt = input().lower()

        found = 0 #did we found the word among our keywords?
        for word in prompt.split(): #split prompt into words
            if found == 1: #first match we found we are good to predict
                break
            for key, value in classes.items(): #look for the word in the dictionary
                if word in value: #if we get a match
                    found = 1 #flag on
                    print(key) #predict the class from the dict
                    break
        if found == 0: #if we didn't find a match, fall back to majority
            print(majority)

    elif choice == "3":
        """
        Testing baseline 2.
        """
        count = 0 #correctly predicted
        incorrect = 0 #incorrectly predicted
        for i, x in enumerate(X_test): #for each test sentence
            found = 0 #did we found the word among our keywords?
            for word in x.split(): #split sentence in words
                if found == 1: #as soon as we found a match for one of our keyboard, go to next sentence
                    break 
                for key, value in classes.items():
                    if word in value: #if the keyword is in the sentence
                        found = 1 #flag on
                        if key == Y_test.iloc[i]: #if the prediction is correct
                            count += 1
                            break
                        else: #if the prediction is incorrect
                            #print("prediction: ", key, "; sentence: ", x, 'incorrect', "; actual class: ", Y_test.iloc[i])
                            incorrect += 1
                            break
            if found == 0: #if after going through the whole dictionary we didn't get a match with one of our keywords
                if majority == Y_test.iloc[i]: #fallback, if it was the majority class
                    count +=1
                else: #if it wasn't
                    #print('fallback failed', x, Y_test.iloc[i])
                    incorrect += 1
        print("Sanity Check", len(X_test), count + incorrect) #sanity check for prediction size and test size
        print(count/len(X_test)) #accuracy
    else:
        break
