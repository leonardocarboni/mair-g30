# -*- coding: utf-8 -*-
"""
Part 1b: modelling and implementing a dialog management system
Group G30 (Giacomo Bais, Leonardo Carboni, Merel de Goede, Merel van den Bos)
"""

from cmath import inf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

states = {
    1: "WELCOME",
    0: "ASK_INFORMATIONS",
    2: "ASK_AREA",
    3: "ASK_FOOD",
    4: "ASK_PRICERANGE",
    5: "RESTAURANT_FOUND",
    6: "RESTAURANT_NOT_FOUND",
    7: "AWAIT_COMMAND",
}

#logistic regression
def train_logistic():
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
    vocab = defaultdict(lambda: len(vocab)) # defaultdict to have indexes for each word
    for sentence in X_train.array: # for each train sentence
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
    return (LogisticRegression(random_state=0, max_iter = 500).fit(train_data, Y_train_reshaped), vocab, LE)

current_state = 1
majority = "inform"

food_types = ['british', 'modern european', 'italian', 'romanian', 'seafood',
              'chinese', 'steakhouse', 'asian oriental', 'french', 'portuguese', 'indian',
              'spanish', 'european', 'vietnamese', 'korean', 'thai', 'moroccan', 'swiss',
              'fusion', 'gastropub', 'tuscan', 'international', 'traditional',
              'mediterranean', 'polynesian', 'african', 'turkish', 'bistro',
              'north american', 'australasian', 'persian', 'jamaican', 'lebanese', 'cuban',
              'japanese', 'catalan']

price_ranges = ['moderate', 'expensive', 'cheap']

areas = ['west', 'east', 'north', 'south', 'centre']

any_utterances = ['don\'t care', 'any', 'whatever']

informations = {'food': None, 'area': None,
                'price': None, 'suitable_list': None}

restaurants = pd.read_csv('restaurant_info.csv')

# suitable_restaurants = pd.DataFrame()


def print_welcome():
    print("Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?")


def extract_class(user_input):
    """
    For each word in the user input, look for that word in the dictionary. If you find it, return the
    key (the class) associated with that word. If you don't find it, return the majority class

    :param user_input: the user's input
    :return: The key of the dictionary.
    """
    user_data = np.zeros(len(vocab))
    for word in user_input.split():
        if word in vocab:
            user_data[vocab[word]] += 1
        else:
            user_data[vocab['NEW_WORD']] += 1
    return LE.inverse_transform(model.predict(user_data.reshape(1,-1)))
def extract_params(ui_split):

    for i, word in enumerate(ui_split):
        # type of food
        if word in food_types:
            informations['food'] = word
        elif word == 'food' and ui_split[i-1] not in price_ranges:
            informations['food'] = ui_split[i-1]

        # price ranges
        if word in price_ranges:
            informations['price'] = word
        elif word == 'moderately':
            informations['price'] = 'moderate'

        # areas
        if word in areas:
            informations['area'] = word


def lookup_restaurants():
    
    food_filter = [1] * len(restaurants)
    price_filter = [1] * len(restaurants)
    area_filter = [1] * len(restaurants)
    
    if informations['food'] is not None:
        food_filter = restaurants['food'] == informations['food']
    if informations['price'] is not None:
        price_filter = restaurants['pricerange'] == informations['price']
    if informations['area'] is not None:
        area_filter = restaurants['area'] == informations['area']
    
    # print the restaurants that match the user's request on the specified parameters
    informations['suitable_list'] = restaurants[food_filter & price_filter & area_filter]

def transition(old_state):
    if old_state == 1:
        print_welcome()
        user_input = input().lower()
        ui_class = extract_class(user_input)
        #print("DEBUG - input class: ", ui_class)

        if ui_class == 'inform':
            ui_split = user_input.split()
            extract_params(ui_split)

            #print("DEBUG - informations: ", informations)

            lookup_restaurants()
            if len(informations['suitable_list']) == 0:  # no restaurant found
                return 6
            if len(informations['suitable_list']) == 1:  # only one restaurant found
                return 5

            if informations['area'] == None:  # area not specified -> ask area
                return 2
            # food type not specified -> ask food type
            elif informations['food'] == None:
                return 3
            # price range not specified -> ask price range
            elif informations['price'] == None:
                return 4
            else:   # all preferences are given
                return 5

        # if the class is not inform, loop back to the beginning
        return 1

    elif old_state == 2:
        print("What area would you like to eat in?")
        user_input = input().lower()
        ui_class = extract_class(user_input)
        #print("DEBUG - input class: ", ui_class)

        if ui_class == 'inform':
            ui_split = user_input.split()
            extract_params(ui_split)

            #print("DEBUG - informations: ", informations)

            lookup_restaurants()  # update the list of suitable restaurants
            # only one restaurant found -> suggest restaurant
            if len(informations['suitable_list']) == 1:
                return 5
            # no restaurants found -> inform user there are no restaurants
            elif len(informations['suitable_list']) == 0:
                return 6
            # more than 1 restaurant found and food type not specified -> ask food type
            elif informations['food'] == None:
                return 3
            # more than 1 restaurant found and price range not specified -> ask price range
            elif informations['price'] == None:
                return 4
            # more than 1 restaurant found and all preferences are specified -> list restaurantsÌ
            else:
                return 5
        return 2

    elif old_state == 3:
        print("What type of food would you like to eat?")
        user_input = input().lower()
        ui_class = extract_class(user_input)
        #print("DEBUG - input class: ", ui_class)

        if ui_class == 'inform':
            ui_split = user_input.split()
            extract_params(ui_split)

            #print("DEBUG - informations: ", informations)

            lookup_restaurants()
            # only one restaurant found -> suggest restaurant
            if len(informations['suitable_list']) == 1:
                return 5
            # no restaurants found -> inform user there are no restaurants
            elif len(informations['suitable_list']) == 0:
                return 6
            # more than 1 restaurant found and price range not specified -> ask price range
            elif informations['price'] == None:
                return 4
            # more than 1 restaurant found and all preferences are specified -> list restaurants
            else:
                return 5
        return 3
    elif old_state == 4:
        print("What price range do you prefer?")
        user_input = input().lower()
        ui_class = extract_class(user_input)
        #print("DEBUG - input class: ", ui_class)

        if ui_class == 'inform':
            ui_split = user_input.split()
            extract_params(ui_split)

            #print("DEBUG - informations: ", informations)

            lookup_restaurants()
            # only one restaurant found -> suggest restaurant
            if len(informations['suitable_list']) == 1:
                return 5
            # no restaurants found -> inform user there are no restaurants
            elif len(informations['suitable_list']) == 0:
                return 6
            # more than 1 restaurant found and all preferences are specified -> list restaurants
            else:
                return 5
        return 4
    elif old_state == 5:
        restaurant = informations['suitable_list'].iloc[0]

        print(f"{restaurant[0]} is a nice place", end="")
        if informations['area'] != None:
            print(f" in the {restaurant[2]} of town", end="")
        if informations['price'] != None:
            print(f" in the {restaurant[1]} price range", end="")
        if informations['food'] != None:
            print(f" serving {restaurant[3]} food", end="")
        print(".")
        return 8
    elif old_state == 6:

        return 8
    return old_state

model, vocab, LE = train_logistic()

while True:
    new_state = transition(current_state)
    current_state = new_state
