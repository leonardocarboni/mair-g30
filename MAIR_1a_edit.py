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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer


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


# Main Loop
while True:

    # User Input
    print("\nChoose what you want to do:")
    print("1. Baseline 1")
    print("2. Baseline 2")
    print("3. Test")
    print("4. Machine Learning 1")
    print("5. Machine Learning 2")
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
        Testing baseline 2.
        """
        count = 0  # correctly predicted
        incorrect = 0  # incorrectly predicted
        for i, x in enumerate(X_test):  # for each test sentence
            found = 0  # did we find the word among our keywords?
            for word in x.split():  # split sentence in words
                if found == 1:  # as soon as we found a match for one of our keyboard, go to next sentence
                    break
                for key, value in classes.items():
                    if word in value:  # if the keyword is in the sentence
                        found = 1  # flag on
                        if key == Y_test.iloc[i]:  # if the prediction is correct
                            count += 1
                            break
                        else:  # if the prediction is incorrect
                            #print("prediction: ", key, "; sentence: ", x, 'incorrect', "; actual class: ", Y_test.iloc[i])
                            incorrect += 1
                            break
            if found == 0:  # if after going through the whole dictionary we didn't get a match with one of our keywords
                # fallback, if it was the majority class
                if majority == Y_test.iloc[i]:
                    count += 1
                else:  # if it wasn't
                    #print('fallback failed', x, Y_test.iloc[i])
                    incorrect += 1
        # sanity check for prediction size and test size
        print("Sanity Check", len(X_test), count + incorrect)
        print(count/len(X_test))  # accuracy
        
    elif choice == "4":
        """
        Machine Learning 1:
        A machine learning system based on Logistic Regression.
        """
        print("\nML 1 - Logistic Regression")
        
        # create the model
        data = df['utterance_content'].array # get whole data to array
        countvec = CountVectorizer() # instantiate vectorizer for bag of word conversion
        cdf = countvec.fit(data) # build the bag of word dictionary
        LE = LabelEncoder() # encode y labels
        Y_train_reshaped = LE.fit_transform(Y_train)
        Y_test_reshaped = LE.fit_transform(Y_test)
        
        train_data = cdf.transform(X_train.array) # convert train data into bag of words
        test_data = cdf.transform(X_test.array) # convert test data into bag of words
        # logistic regressor
        LR = LogisticRegression(random_state=0).fit(train_data.toarray(), Y_train_reshaped)
        LR.predict(test_data.toarray())
        print("R^2 is: ", round(LR.score(test_data.toarray(), Y_test_reshaped), 4)) # R^2 score
        
    elif choice == "5":
        """
        Machine Learning 2:
        A machine learning system based on MLPC Classifier Neural Network.
        """
        print("\nML 2 - Neural Network")
        
        # create the model
        LE = LabelEncoder()
        Y_train_reshaped = LE.fit_transform(Y_train).reshape(-1, 1)
        Y_test_reshaped = LE.fit_transform(Y_test).reshape(-1, 1)
        
        NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(Y_train_reshaped, X_train)
        NN.predict(Y_test_reshaped)
        print(round(NN.score(Y_test_reshaped,X_test), 4))
        
        
    else:
        break

