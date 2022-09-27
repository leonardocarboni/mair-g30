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
from functools import lru_cache

prev_state = 1

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


# logistic regression
## LOOK UP PICKLE FOR PRE-TRAINING ##

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
    # defaultdict to have indexes for each word
    vocab = defaultdict(lambda: len(vocab))
    for sentence in df['utterance_content'].array:  # for each train sentence
        for word in sentence.split():  # for each word
            vocab[word]  # build the vocab with progressive indexes

    vocab['NEW_WORD']  # special entry for unseen words
    train_data = np.zeros((len(X_train), len(vocab)))  # bag of word train
    for i, sentence in enumerate(X_train.array):
        for word in sentence.split():
            if word in vocab:
                train_data[i][vocab[word]] += 1  # count words occurances
            else:  # in train this should not occur
                train_data[i][vocab['NEW_WORD']] += 1  # count unseen words

    LE = LabelEncoder()  # encode y labels
    Y_train_reshaped = LE.fit_transform(Y_train)
    return (LogisticRegression(random_state=0, max_iter=500).fit(train_data, Y_train_reshaped), vocab, LE)


model, vocab, LE = train_logistic()


# suitable_restaurants = pd.DataFrame()
def lev_dist(a, b):
    '''
    This function will calculate the levenshtein distance between two input
    strings a and b

    params:
        a (String) : The first string you want to compare
        b (String) : The second string you want to compare

    returns:
        This function will return the distnace between string a and b.

    example:
        a = 'stamp'
        b = 'stomp'
        lev_dist(a,b)
        >> 1.0
    '''

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

# change each word to its closest match  among our keywords according to lev distance


def change_to_lev(user_word):
    if len(user_word) > 3:  # only change if the word has 4 or more letters (if 3, eat -> east and it's annoying, with baseline2 it's easier to just treat it as special case)
        min_dist = inf
        new_word = None
        # check the food types, price ranges and areas for information extraction purposes
        for food_type in food_types:
            if min_dist > lev_dist(user_word, food_type) and lev_dist(user_word, food_type) <= 1:
                new_word = food_type
                min_dist = lev_dist(user_word, food_type)
        for price in price_ranges:
            if min_dist > lev_dist(user_word, price) and lev_dist(user_word, price) <= 1:
                new_word = price
                min_dist = lev_dist(user_word, price)
        for area in areas:
            if min_dist > lev_dist(user_word, area) and lev_dist(user_word, area) <= 1:
                new_word = area
                min_dist = lev_dist(user_word, area)
        if new_word is not None:
            return new_word
    return user_word


# type of restaurant: ([true inference list], [false inference list])
rules = {
    "romantic": (["long stay"], ["busy"]),
    "childern": (["short stay"], ["long stay"]),
    "assigned seats": (["busy"], ["not busy"]),
    "touristic": (["cheap", "good food"], ["romanian"]),
}


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
        word = change_to_lev(word)
        if word in vocab:
            user_data[vocab[word]] += 1
        else:
            user_data[vocab['NEW_WORD']] += 1
    return LE.inverse_transform(model.predict(user_data.reshape(1, -1)))


def extract_params(ui_split):

    for i, word in enumerate(ui_split):
        word = change_to_lev(word)
        # type of food
        if word in food_types:
            informations['food'] = word
        elif word == 'food' and change_to_lev(ui_split[i-1]) not in price_ranges:
            informations['food'] = change_to_lev(ui_split[i-1])
        elif word == 'asian':
            informations['food'] = 'asian oriental'

        # price ranges
        if word in price_ranges:
            informations['price'] = word
        elif word == 'moderately':
            informations['price'] = 'moderate'

        # areas
        if word in areas:
            informations['area'] = word
        elif word == 'center':
            informations['area'] = 'centre'


def lookup_restaurants():

    food_filter = [True] * len(restaurants)
    price_filter = [True] * len(restaurants)
    area_filter = [True] * len(restaurants)

    if informations['food'] is not None:
        food_filter = (restaurants['food'] ==
                       informations['food']).values.tolist()
    if informations['price'] is not None:
        price_filter = (restaurants['pricerange'] ==
                        informations['price']).values.tolist()
    if informations['area'] is not None:
        area_filter = (restaurants['area'] ==
                       informations['area']).values.tolist()

    final_filter = [all(i)
                    for i in zip(food_filter, price_filter, area_filter)]

    # print the restaurants that match the user's request on the specified parameters
    informations['suitable_list'] = restaurants[final_filter]


def transition(current_state):
    if current_state == 1:
        print_welcome()
        user_input = input().lower()
        ui_class = extract_class(user_input)
        print("DEBUG - input class: ", ui_class)

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
            else:  # all preferences are given and more than 1 restaurant found
                return 91
        elif ui_class == 'bye':
            return 12
        elif ui_class == 'repeat':
            return current_state
        # if the class is not inform, loop back to the beginning
        return current_state

    elif current_state == 2:
        print("What area would you like to eat in?")
        user_input = input().lower()
        ui_class = extract_class(user_input)
        print("DEBUG - input class: ", ui_class)

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
                return 91
        elif ui_class == 'bye':
            return 12
        elif ui_class == 'repeat':
            return current_state
        return current_state

    elif current_state == 3:
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
                return 91
        elif ui_class == 'bye':
            return 12
        elif ui_class == 'repeat':
            return current_state
        return current_state
    elif current_state == 4:
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
                return 91
        elif ui_class == 'bye':
            return 12
        elif ui_class == 'repeat':
            return current_state
        return current_state
    elif current_state == 91:
        print("Do you have additional requirements?")
        return 5
    elif current_state == 5:
        restaurant = informations['suitable_list'].iloc[0]

        print(f"{restaurant[0]} is a nice place", end="")
        if informations['area'] != None:
            print(f" in the {restaurant[2]} of town", end="")
        if informations['price'] != None:
            print(f" in the {restaurant[1]} price range", end="")
        if informations['food'] != None:
            print(f" serving {restaurant[3]} food", end="")
        print(".")

        return 7

    elif current_state == 6:
        print("I'm sorry but there is no restaurant", end="")
        if informations['area'] != None:
            print(f" in the {informations['area']} of town", end="")
            informations['area'] = None
        if informations['price'] != None:
            print(f" in the {informations['price']} price range", end="")
            informations['price'] = None
        if informations['food'] != None:
            print(f" serving {informations['food']} food", end="")
            informations['food'] = None
        print(".")

        return 7

    elif current_state == 7:
        user_input = input().lower()
        ui_class = extract_class(user_input)

        print("DEBUG - input class: ", ui_class)

        if ui_class == 'inform':
            ui_split = user_input.split()

            extract_params(ui_split)

            print("DEBUG - informations: ", informations)

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

        elif ui_class == 'request':
            ui_split = user_input.split()
            for i, word in enumerate(ui_split):
                if word == 'postcode' or word == 'post':
                    return 8
                if word == 'address':
                    print('help me 2.0')
                    return 9
                if word == 'phone' or word == 'phonenumber':
                    return 10
        elif ui_class == 'reqalts':
            # If there is another restaurant, recommend the different restaurant
            if len(informations['suitable_list']) > 1:
                informations['suitable_list'] = informations['suitable_list'][1:]
                return 5
            # If there is no other restaurant, tell the user
            return 11
        # reqmore not implemented bc it doesn't make sense
        elif ui_class == 'repeat':
            return prev_state
        elif ui_class == 'bye' or ui_class == 'thankyou':
            return 12
        return 7

    elif current_state == 8:
        restaurant = informations['suitable_list'].iloc[0]
        print(f"The post code of {restaurant[0]} is {restaurant[6]}.")
        return 7

    elif current_state == 9:
        restaurant = informations['suitable_list'].iloc[0]
        print(f"The address of {restaurant[0]} is {restaurant[5]}.")
        return 7

    elif current_state == 10:
        restaurant = informations['suitable_list'].iloc[0]
        print(f"The phone number of {restaurant[0]} is {restaurant[4]}.")
        return 7

    elif current_state == 11:
        print("Sorry but there is no other restaurant", end="")
        if informations['area'] != None:
            print(f" in the {restaurant[2]} of town", end="")
        if informations['price'] != None:
            print(f" in the {restaurant[1]} price range", end="")
        if informations['food'] != None:
            print(f" serving {restaurant[3]} food", end="")
        print(".")
        return 7

    elif current_state == 12:
        print("bye")
        return -1
    return current_state


while True:
    new_state = transition(prev_state)
    if new_state == -1:
        break
    prev_state = new_state
