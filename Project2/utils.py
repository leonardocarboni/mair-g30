
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from collections import defaultdict
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from functools import lru_cache

classes = {
    'ack': ['kay', 'okay', "fine", 'great', 'good'],
    'bye': ['bye', 'goodbye', 'see', 'talk'],
    'affirm': ['yes', 'yeah', 'yep', 'right', 'indeed'],
    'confirm': ['true', 'correct'],
    'deny': ['don\'t', 'cannot', 'cant', 'can\'t', 'nope', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor', 'never', 'none', 'hardly'],
    'hello': ['hello', 'hi', 'hey', 'morning', 'afternoon', 'evening'],
    'inform': ['eat', 'look', 'looking', 'search', 'find', 'want', 'need', 'require', 'requirement', 'west', 'east', 'north', 'south', 'restaurant', 'food', 'town'],
    'negate': ['no', 'nope', 'not', 'never', 'none', 'nothing', 'nah'],
    'null': ['cough', 'clear', 'laugh', 'sigh', 'sniff', 'noise', 'sil', 'unintelligible'],
    'repeat': ['again', 'repeat'],
    'reqalts': ['alternatives', 'other', 'another', 'different', 'else', 'other'],
    'reqmore': ['more'],
    'request': ['where' '?', 'train', 'taxi', 'plane', 'phone', 'how', 'why', 'number', 'price', 'post', 'code', 'postcode', 'address', 'phonenumber'],
    'restart': ['start', 'restart', 'again', 'beginning'],
    'thankyou': ['thank', 'thanks', 'thankyou'],
}
d = pd.read_csv('Data\dialog_acts.dat', header=None)
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

# getting the stopword list from nltk
sw = set(stopwords.words('english'))

#Function that train a logistic regression model given data
def train_logistic(data):
    print('Training...')
    vocab = defaultdict(lambda: len(vocab)) # defaultdict to have indexes for each word
    for sentence in data['utterance_content'].array: # for each train sentence
        for word in sentence.split(): # for each word
            if word not in sw:
                vocab[word] # build the vocab with progressive indexes
            
    vocab['NEW_WORD'] # special entry for unseen words
    train_data = np.zeros((len(X_train), len(vocab))) # bag of word train
    for i, sentence in enumerate(X_train.array):
        for word in sentence.split():
            if word not in sw:
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

# Function to train a decision tree
def train_tree(data):
    print('Training...')
    # much of the same as the previous model
    vocab = defaultdict(lambda: len(vocab))
    for sentence in data['utterance_content'].array:
        for word in sentence.split():
            if word not in sw:
                vocab[word]
        
    vocab['NEW_WORD']
    train_data = np.zeros((len(X_train), len(vocab)))
    for i, sentence in enumerate(X_train.array):
        for word in sentence.split():
            if word not in sw:
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

def train_MLP(data):
    print('Training...')
    vocab = defaultdict(lambda: len(vocab)) # defaultdict to have indexes for each word
    for sentence in data['utterance_content'].array: # for each train sentence
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
    clf = MLPClassifier(random_state=1).fit(train_data, Y_train_reshaped)
    return clf, LE, vocab

# extract majority class from data
majority = Y_train.mode()[0]

# Function that checks the lower or upper casing of system output
def caps_check_print(text, caps_lock, end="\n"):
    if caps_lock:
        print(text.upper(), end=end)
    else:
        print(text, end=end)

#Function that given two strings, returns their Levenshtein edit distance
def lev_dist(a, b):
    
    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):

        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),      # insert character
            min_dist(s1 + 1, s2),      # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)