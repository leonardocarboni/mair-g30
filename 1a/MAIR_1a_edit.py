# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:15:01 2022

@author: bais_
"""


"""
Part 1a: text classification
Group G30 (Giacomo Bais, Leonardo Carboni, Merel de Goede, Merel van den Bos)
"""


# Creating the dataframe
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from collections import defaultdict

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
    'ack': ['kay', 'okay', "fine", 'great', 'good'],
    'affirm': ['yes', 'yeah', 'yep', 'right', 'indeed'],
    'bye': ['bye', 'goodbye', 'see', 'talk'],
    'confirm': ['true', 'correct'],
    'deny': ['don\'t', 'cannot', 'cant', 'can\'t', 'no', 'nope', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor', 'never', 'none', 'hardly'],
    'hello': ['hello', 'hi', 'hey', 'morning', 'afternoon', 'evening'],
    'inform': ['look', 'looking', 'search', 'find', 'want', 'need', 'require', 'requirement', 'west', 'east', 'north', 'south', 'restaurant', 'food', 'town'],
    'negate': ['no', 'nope', 'not', 'never', 'none', 'nothing', 'nah'],
    'null': ['cough', 'clear', 'laugh', 'sigh', 'sniff', 'noise', 'sil', 'unintelligible'],
    'repeat': ['again', 'repeat'],
    'reqalts': ['about', 'alternatives', 'other', 'another', 'different', 'else', 'other'],
    'reqmore': ['more'],
    'request': ['whats', 'what\'s', 'restaurant', 'where' '?', 'what', 'train', 'taxi', 'plane', 'phone', 'how', 'why', 'can', 'number', 'price'],
    'restart': ['start', 'restart', 'again', 'beginning'],
    'thankyou': ['thank', 'thanks', 'thankyou'],
}

# function to train a logistic classifier
def train_logistic():
    print('Training...')
    vocab = defaultdict(lambda: len(vocab)) # defaultdict to have indexes for each word
    for sentence in df['utterance_content'].array: # for each train sentence
        for word in sentence.split(): # for each word
            vocab[word] # build the vocab with progressive indexes
            
    vocab['NEW_WORD'] # special entry for unseen words
    train_data = np.zeros((len(X_train), len(vocab))) # bag of word train
    for i, sentence in enumerate(X_train.array):
        for word in sentence.split():
            if word in vocab:
                train_data[i][vocab[word]] += 1 # count words occurances 
            else: # in train this should not occur
                train_data[i][vocab['NEW_WORD']] += 1 # count unseen words
            
    LE = LabelEncoder() # encode y labels
    Y_train_reshaped = LE.fit_transform(Y_train)
    Y_test_reshaped = LE.fit_transform(Y_test)
            
            # logistic regressor
    LR = LogisticRegression(random_state=0, max_iter = 500).fit(train_data, Y_train_reshaped)
    return LR, LE, vocab

# function to train a decision tree
def train_tree():
    print('Training...')
    # much of the same as the previous model
    vocab = defaultdict(lambda: len(vocab))
    for sentence in df['utterance_content'].array:
        for word in sentence.split():
            vocab[word]
        
    vocab['NEW_WORD']
    train_data = np.zeros((len(X_train), len(vocab)))
    for i, sentence in enumerate(X_train.array):
        for word in sentence.split():
            if word in vocab:
                train_data[i][vocab[word]] += 1
            else:
                train_data[i][vocab['NEW_WORD']] += 1
        
    LE = LabelEncoder() # encode y labels
    Y_train_reshaped = LE.fit_transform(Y_train)
    Y_test_reshaped = LE.fit_transform(Y_test)
        
    # decision tree classifier
    clf = DecisionTreeClassifier(random_state=0).fit(train_data, Y_train_reshaped)
    return clf, LE, vocab

# Main Loop
while True:

    # User Input
    print("\nChoose what you want to do:")
    print("1. Baseline 1")
    print("2. Baseline 2")
    print("3. Logistic Regression")
    print("4. Decision Tree Classifier")
    print("5. Evaluations")
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
        print(f'Predicted class: {majority}')

    elif choice == "2":
        """
        Baseline 2:
        A baseline rule-based system based on keyword matching.
        """
        print("\nBaseline 2 - Write a sentence or a word")

        # Reading input and converting it in lower case
        prompt = input().lower()

        found = 0  # did we find the word among our keywords?
        for word in prompt.split():  # split prompt into words
            if found == 1:  # first match we found we are good to predict
                break
            for key, value in classes.items():  # look for the word in the dictionary
                if word in value:  # if we get a match
                    found = 1  # flag on
                    print(key)  # predict the class from the dict
                    break
        if found == 0:  # if we didn't find a match, fall back to majority
            print(majority)
        
    elif choice == "3":
        """
        Machine Learning 1:
        A machine learning system based on Logistic Regression.
        """
        print("\nML 1 - Logistic Regression")
        
        # create the model, the label encoder and vocab
        LR, LE, vocab = train_logistic()
        
        print("Now you can write a sentence or a word to test the model.")

        # Reading input and converting it in lower case
        prompt = input().lower()
        print("The predicted class for the sentence is ")
        
        # creating bag of words for user input
        user_data = np.zeros(len(vocab))
        for word in prompt.split():
            if word in vocab:
                user_data[vocab[word]] += 1
            else:
                user_data[vocab['NEW_WORD']] += 1
        print(LE.inverse_transform(LR.predict(user_data.reshape(1,-1)))) # predict class and print
        
        
    elif choice == "4":
        """
        Machine Learning 2:
        A machine learning system based on a decision tree classifier.
        """
        print("\nML 2 - Decision Tree")
        
        clf, LE, vocab = train_tree()
        
        print("Now you can write a sentence or a word to test the model.")

        # Reading input and converting it in lower case
        prompt = input().lower()
        print("The predicted class for the sentence is ")
        
        user_data = np.zeros(len(vocab))
        for word in prompt.split():
            if word in vocab:
                user_data[vocab[word]] += 1
            else:
                user_data[vocab['NEW_WORD']] += 1
        print(LE.inverse_transform(clf.predict(user_data.reshape(1,-1))))
                
    elif choice == "5": 
        """
        Testing baseline 2.
        """
        print("\nChoose what you want to evaluate:")
        print("1. Baseline 1")
        print("2. Baseline 2")
        print("3. Logistic Regression")
        print("4. Decision Tree Classifier")

        eval_choice = input("Enter your choice: ")
        if eval_choice == '1':

            # evaluating baseline 1
            evaluation = classification_report(Y_test, [majority]*len(Y_test), zero_division = 0)
            print()
            print('Baseline 1 evaluation:')
            print(evaluation)

        elif eval_choice == '2':
            # evaluating baseline 2
            test_preds = []
            for i, x in enumerate(X_test):  # for each test sentence
                found = 0  # did we find the word among our keywords?
                for word in x.split():  # split sentence in words
                    if found == 1:  # as soon as we found a match for one of our keyboard, go to next sentence
                        break
                    for key, value in classes.items():
                        if word in value:  # if the keyword is in the sentence
                            found = 1  # flag on
                            test_preds.append(key)
                            break
                if found == 0:  # if after going through the whole dictionary we didn't get a match with one of our keywords
                    # fallback, if it was the majority class
                    test_preds.append(majority)
            # sanity check for prediction size and test size
            print("Sanity Check", len(X_test), len(test_preds))
            evaluation = classification_report(Y_test, test_preds, zero_division = 0)
            print()
            print('Baseline 2 evaluation:')
            print(evaluation)

        elif eval_choice == '3':
            # building and evaluating Logistic Regressor
            LR, LE, vocab = train_logistic()
            test_data = np.zeros((len(X_test), len(vocab)))
            for i, sentence in enumerate(X_test.array):
                for word in sentence.split():
                    if word in vocab:
                        test_data[i][vocab[word]] += 1
                    else:
                        test_data[i][vocab['NEW_WORD']] += 1
            preds = LE.inverse_transform(LR.predict(test_data))
            evaluation = classification_report(Y_test, preds, zero_division = 0)
            print()
            print('Logistic Regression evaluation:')
            print(evaluation)

        elif eval_choice == '4':
            # building and evaluating Decision Tree
            clf, LE, vocab = train_tree()
            test_data = np.zeros((len(X_test), len(vocab)))
            for i, sentence in enumerate(X_test.array):
                for word in sentence.split():
                    if word in vocab:
                        test_data[i][vocab[word]] += 1
                    else:
                        test_data[i][vocab['NEW_WORD']] += 1
            preds = LE.inverse_transform(clf.predict(test_data))
            evaluation = classification_report(Y_test, preds, zero_division = 0)
            print()
            print('Decision Tree evaluation:')
            print(evaluation)
        
    else:
        break

