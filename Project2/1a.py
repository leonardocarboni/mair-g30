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
import numpy as np
import utils

from sklearn.metrics import classification_report


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
        print(f'Predicted class: {utils.majority}')

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
            for key, value in utils.classes.items():  # look for the word in the dictionary
                if word in value:  # if we get a match
                    found = 1  # flag on
                    print(key)  # predict the class from the dict
                    break
        if found == 0:  # if we didn't find a match, fall back to majority
            print(utils.majority)
        
    elif choice == "3":
        """
        Machine Learning 1:
        A machine learning system based on Logistic Regression.
        """
        print("\nML 1 - Logistic Regression")
        
        # create the model, the label encoder and vocab
        LR, LE, vocab = utils.train_logistic(utils.df)
        
        print("Now you can write a sentence or a word to test the model.")

        # Reading input and converting it in lower case
        prompt = input().lower()
        print("The predicted class for the sentence is ")
        
        # creating bag of words for user input
        user_data = np.zeros(len(vocab))
        for word in prompt.split():
            if word not in utils.sw:
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
        
        clf, LE, vocab = utils.train_tree(utils.df)
        
        print("Now you can write a sentence or a word to test the model.")

        # Reading input and converting it in lower case
        prompt = input().lower()
        print("The predicted class for the sentence is ")
        
        user_data = np.zeros(len(vocab))
        for word in prompt.split():
            if word not in utils.sw:
                if word in vocab:
                    user_data[vocab[word]] += 1
                else:
                    user_data[vocab['NEW_WORD']] += 1
        print(LE.inverse_transform(clf.predict(user_data.reshape(1,-1))))
                
    elif choice == "5": 
        # Evaluating models

        print("\nChoose what you want to evaluate:")
        print("1. Baseline 1")
        print("2. Baseline 2")
        print("3. Logistic Regression")
        print("4. Decision Tree Classifier")

        eval_choice = input("Enter your choice: ")
        if eval_choice == '1':

            # evaluating baseline 1
            evaluation = classification_report(utils.Y_test, [utils.majority]*len(utils.Y_test), zero_division = 0)
            print()
            print('Baseline 1 evaluation:')
            print(evaluation)

        elif eval_choice == '2':
            # evaluating baseline 2
            test_preds = []
            for i, x in enumerate(utils.X_test):  # for each test sentence
                found = 0  # did we find the word among our keywords?
                for word in x.split():  # split sentence in words
                    if found == 1:  # as soon as we found a match for one of our keyboard, go to next sentence
                        break
                    for key, value in utils.classes.items():
                        if word in value:  # if the keyword is in the sentence
                            found = 1  # flag on
                            test_preds.append(key)
                            break
                if found == 0:  # if after going through the whole dictionary we didn't get a match with one of our keywords
                    # fallback, if it was the majority class
                    test_preds.append(utils.majority)
            evaluation = classification_report(utils.Y_test, test_preds, zero_division = 0)
            print()
            print('Baseline 2 evaluation:')
            print(evaluation)

        elif eval_choice == '3':
            # building and evaluating Logistic Regressor
            LR, LE, vocab = utils.train_logistic(utils.df)
            test_data = np.zeros((len(utils.X_test), len(vocab)))
            for i, sentence in enumerate(utils.X_test.array):
                for word in sentence.split():
                    if word not in utils.sw:
                        if word in vocab:
                            test_data[i][vocab[word]] += 1
                        else:
                            test_data[i][vocab['NEW_WORD']] += 1
            preds = LE.inverse_transform(LR.predict(test_data))
            evaluation = classification_report(utils.Y_test, preds, zero_division = 0)
            print()
            print('Logistic Regression evaluation:')
            print(evaluation)

        elif eval_choice == '4':
            # building and evaluating Decision Tree
            clf, LE, vocab = utils.train_tree(utils.df)
            test_data = np.zeros((len(utils.X_test), len(vocab)))
            for i, sentence in enumerate(utils.X_test.array):
                for word in sentence.split():
                    if word not in utils.sw:
                        if word in vocab:
                            test_data[i][vocab[word]] += 1
                        else:
                            test_data[i][vocab['NEW_WORD']] += 1
            preds = LE.inverse_transform(clf.predict(test_data))
            evaluation = classification_report(utils.Y_test, preds, zero_division = 0)
            print()
            print('Decision Tree evaluation:')
            print(evaluation)
        
    else:
        break

