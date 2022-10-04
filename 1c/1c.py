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
from enum import Enum


class State(Enum):
    WELCOME = 1
    ASK_AREA = 2
    ASK_FOOD = 3
    ASK_PRICE = 4
    ASK_REQUIREMENTS = 5
    RESTAURANT_FOUND = 6
    RESTAURANT_NOT_FOUND = 7
    AWAIT_COMMAND = 8
    PRINT_POSTCODE = 9
    PRINT_ADDRESS = 10
    PRINT_PHONENUMBER = 11
    NO_OTHER_RESTAURANTS = 12
    BYE = 13
    KILL = -1

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

prev_state = State.WELCOME

majority = "inform"

chosen_model = 1 # logistic regression default

auto_correction = True # use levenshtein edit distance or not

caps_lock = False

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
                'price': None, 'suitable_list': None, 'extra': None}

restaurants = pd.read_csv('restaurant_info2.csv')

# type of restaurant: ([true inference list], [false inference list])
rules = {
    "romantic": (["long stay"], ["busy"]),
    "children": (["short stay"], ["long stay"]),
    "assigned": (["busy"], ["not busy"]),
    "touristic": (["cheap", "good food"], ["romanian"]),
}


def caps_check_print(text, end="\n"):
    if caps_lock:
        print(text.upper(), end=end)
    else:
        print(text)

# logistic regression
## LOOK UP PICKLE FOR PRE-TRAINING ##


def train_logistic():
    """
    It reads the data, splits it into train and test, builds a vocabulary, and trains a logistic
    regression model
    :return: A tuple containing the trained model, the vocabulary and the label encoder.
    """
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




def lev_dist(a, b):
    '''
    This function will calculate the levenshtein distance between two input
    strings a and b
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


def change_to_lev(user_word):
    """
    Change each word to its closest match  among our keywords according to lev distance

    :param user_word: the word that the user said
    :return: the word that is closest to the user_word in terms of Levenshtein distance.
    """
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


def print_welcome():
    caps_check_print("Hello, welcome to the MAIR G30 restaurant system? You can ask for restaurants by area, price range or food type. You can also add additional requirements among: Romantic; Touristic; Children; Assigned seats. How may I help you?")


def extract_class(user_input):
    """
    It takes in a string, and returns the class that the model predicts

    :param user_input: The user's input
    :return: The class of the input
    """
    if chosen_model == 1: #logistic regression
        user_data = np.zeros(len(vocab))
        for word in user_input.split():
            if auto_correction:
                word = change_to_lev(word)
            if word in vocab:
                user_data[vocab[word]] += 1
            else:
                user_data[vocab['NEW_WORD']] += 1
        return LE.inverse_transform(model.predict(user_data.reshape(1, -1)))
    else : # baseline 2
        for word in user_input.split():  # split prompt into words
            if auto_correction:
                word = change_to_lev(word)
            for key, value in classes.items():  # look for the word in the dictionary
                if word in value:  # if we get a match
                    return(key)  # predict the class from the dict
        return(majority)

def extract_params(ui_split):
    """
    It takes the user input and checks if it contains any of the keywords in the dictionaries. If it
    does, it adds the keyword to the informations dictionary

    :param ui_split: the user input split into a list of words
    """
    for i, word in enumerate(ui_split):
        prev_word = None
        if auto_correction:
            word = change_to_lev(word)
            if i != 0:
                prev_word = change_to_lev(ui_split[i-1])
        elif i != 0:
            prev_word = ui_split[i-1]
        # type of food
        if word in food_types:
            informations['food'] = word
        elif word == 'food' and prev_word is not None and prev_word not in price_ranges:
            informations['food'] = prev_word
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


def manage_requirements():
    """
    It takes the extra information and filters the suitable list based on that information
    :return: a string that explains why the restaurant is suitable for the extra requirement.
    """
    if informations['extra'] == 'romantic':
        informations['suitable_list'] = informations['suitable_list'][informations['suitable_list']
                                                                      ['stay_length'] == 'long stay']
        return "The restaurant is romantic because it allows you to stay for a long time."
    if informations['extra'] == 'children':
        informations['suitable_list'] = informations['suitable_list'][informations['suitable_list']
                                                                      ['stay_length'] == 'short stay']
        return "The restaurant is for children because it allows you to stay for a short time."
    if informations['extra'] == 'assigned':
        informations['suitable_list'] = informations['suitable_list'][informations['suitable_list']
                                                                      ['crowdedness'] == 'busy']
        return "The restaurant allows for assigned seats because it is usually busy."
    if informations['extra'] == 'touristic':
        informations['suitable_list'] = informations['suitable_list'][(informations['suitable_list']['pricerange'] == 'cheap') & (
            informations['suitable_list']['food_quality'] == 'good food')]
        return "The restaurant is touristic because it is cheap and it serves good food."


def lookup_restaurants():
    """
    It filters the restaurants dataframe based on the user's input and stores the filtered dataframe in
    the informations dictionary
    """
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
    if current_state == State.WELCOME:
        print_welcome()
        user_input = input().lower()
        ui_class = extract_class(user_input)

        if ui_class == 'inform' or ui_class == 'deny':
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()

            if len(informations['suitable_list']) == 0:  # no restaurant found
                return State.RESTAURANT_NOT_FOUND
            if len(informations['suitable_list']) == 1:  # only one restaurant found
                return State.RESTAURANT_FOUND

            if informations['area'] == None:  # area not specified -> ask area
                return State.ASK_AREA
            # food type not specified -> ask food type
            elif informations['food'] == None:
                return State.ASK_FOOD
            # price range not specified -> ask price range
            elif informations['price'] == None:
                return State.ASK_PRICE
            else:  # all preferences are given and more than 1 restaurant found
                return State.ASK_REQUIREMENTS

        elif ui_class == 'bye':
            return State.BYE

        elif ui_class == 'repeat':
            return current_state

        elif ui_class == 'restart':
            for info in informations:
                informations[info] = None
            return State.WELCOME
        # if the class is not inform, loop back to the beginning
        return current_state

    elif current_state == State.ASK_AREA:
        caps_check_print("What area would you like to eat in?")
        user_input = input().lower()
        ui_class = extract_class(user_input)

        if ui_class == 'inform' or ui_class == 'deny':
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()  # update the list of suitable restaurants

            # only one restaurant found -> suggest restaurant
            if len(informations['suitable_list']) == 1:
                return State.RESTAURANT_FOUND
            # no restaurants found -> inform user there are no restaurants
            elif len(informations['suitable_list']) == 0:
                return State.RESTAURANT_NOT_FOUND
            # more than 1 restaurant found and food type not specified -> ask food type
            elif informations['food'] == None:
                return State.ASK_FOOD
            # more than 1 restaurant found and price range not specified -> ask price range
            elif informations['price'] == None:
                return State.ASK_PRICE
            # more than 1 restaurant found and all preferences are specified -> list restaurantsÌ
            else:
                return State.ASK_REQUIREMENTS

        elif ui_class == 'bye':
            return State.BYE
        elif ui_class == 'repeat':
            return current_state
        elif ui_class == 'restart':
            for info in informations:
                informations[info] = None
            return State.WELCOME
        return current_state

    elif current_state == State.ASK_FOOD:
        caps_check_print("What type of food would you like to eat?")
        user_input = input().lower()
        ui_class = extract_class(user_input)

        if ui_class == 'inform' or ui_class == 'deny':
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()

            # only one restaurant found -> suggest restaurant
            if len(informations['suitable_list']) == 1:
                return State.RESTAURANT_FOUND
            # no restaurants found -> inform user there are no restaurants
            elif len(informations['suitable_list']) == 0:
                return State.RESTAURANT_NOT_FOUND
            # more than 1 restaurant found and price range not specified -> ask price range
            elif informations['price'] == None:
                return State.ASK_PRICE
            # more than 1 restaurant found and all preferences are specified -> list restaurants
            else:
                return State.ASK_REQUIREMENTS

        elif ui_class == 'bye':
            return State.BYE
        elif ui_class == 'repeat':
            return current_state
        elif ui_class == 'restart':
            for info in informations:
                informations[info] = None
            return State.WELCOME
        return current_state

    elif current_state == State.ASK_PRICE:
        caps_check_print("What price range do you prefer?")
        user_input = input().lower()
        ui_class = extract_class(user_input)

        if ui_class == 'inform' or ui_class == 'deny':
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()

            # only one restaurant found -> suggest restaurant
            if len(informations['suitable_list']) == 1:
                return State.RESTAURANT_FOUND
            # no restaurants found -> inform user there are no restaurants
            elif len(informations['suitable_list']) == 0:
                return State.RESTAURANT_NOT_FOUND
            # more than 1 restaurant found and all preferences are specified -> list restaurants
            else:
                return State.ASK_REQUIREMENTS

        elif ui_class == 'bye':
            return State.BYE
        elif ui_class == 'repeat':
            return current_state
        elif ui_class == 'restart':
            for info in informations:
                informations[info] = None
            return State.WELCOME
        return current_state
    elif current_state == State.ASK_REQUIREMENTS:
        caps_check_print("Do you have additional requirements?")
        return State.AWAIT_COMMAND

    elif current_state == State.RESTAURANT_FOUND:
        req_string = manage_requirements()

        if len(informations['suitable_list']) == 0:
            return State.RESTAURANT_NOT_FOUND

        restaurant = informations['suitable_list'].iloc[0]
        caps_check_print(f"{restaurant[0]} is a nice place", end="")

        if informations['area'] != None:
            caps_check_print(f" in the {restaurant[2]} of town", end="")
        if informations['price'] != None:
            caps_check_print(f" in the {restaurant[1]} price range", end="")
        if informations['food'] != None:
            caps_check_print(f" serving {restaurant[3]} food", end="")
        caps_check_print(".")

        if req_string != None:
            caps_check_print(req_string)

        return State.AWAIT_COMMAND

    elif current_state == State.RESTAURANT_NOT_FOUND:
        caps_check_print("I'm sorry but there is no restaurant", end="")
        if informations['area'] != None:
            caps_check_print(f" in the {informations['area']} of town", end="")
        if informations['price'] != None:
            caps_check_print(f" in the {informations['price']} price range", end="")
        if informations['food'] != None:
            caps_check_print(f" serving {informations['food']} food", end="")

        # if there was a requirements but no restaurants met that requirement
        if informations['extra'] != None:
            # check what requirement was asked an build the answer string
            for word in informations['extra'].split():
                if word == 'romantic' or word == 'touristic':
                    caps_check_print(f' that is also {word}', end="")
                if word == 'children':
                    caps_check_print(f' that is also for {word}', end="")
                if word == 'assigned':
                    caps_check_print(f' that also allows for {word} seats', end="")
            # if there is no restaurant given the requirements, reset string for inference in case of future suggestions
            informations['extra'] = None
        caps_check_print(".")

        return State.AWAIT_COMMAND

    elif current_state == State.AWAIT_COMMAND:
        user_input = input().lower()
        ui_class = extract_class(user_input)

        if ui_class == 'inform':
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()

            if len(informations['suitable_list']) == 0:  # no restaurant found
                return State.RESTAURANT_NOT_FOUND
            if len(informations['suitable_list']) == 1:  # only one restaurant found
                return State.RESTAURANT_FOUND

            for word in ui_split:
                if word in rules.keys():
                    informations['extra'] = word

            # if a requirement was given suggest the restaurant
            if informations['extra'] != None:
                return State.RESTAURANT_FOUND
            # if a requirement wasn't given, user is trying to change the main 3 infos and we go back
            # these 3 checks are probably useless #
            elif informations['area'] == None:  # area not specified -> ask area
                return State.ASK_AREA
            # # food type not specified -> ask food type
            elif informations['food'] == None:
                return State.ASK_FOOD
            # # price range not specified -> ask price range
            elif informations['price'] == None:
                return State.ASK_PRICE
            else:  # if we have all informations but a preference wasn't given, then we ask for them
                return State.ASK_REQUIREMENTS

        elif ui_class == 'request':
            ui_split = user_input.split()

            for word in ui_split:
                if word == 'postcode' or word == 'post':
                    return State.PRINT_POSTCODE
                elif word == 'address':
                    return State.PRINT_ADDRESS
                elif word == 'phone' or word == 'phonenumber':
                    return State.PRINT_PHONENUMBER

        elif ui_class == 'reqalts':
            # If there is another restaurant, recommend the different restaurant
            if len(informations['suitable_list']) > 1:
                informations['suitable_list'] = informations['suitable_list'][1:]
                return State.RESTAURANT_FOUND
            # If there is no other restaurant, tell the user
            return State.NO_OTHER_RESTAURANTS

        # reqmore not implemented bc it doesn't make sense

        elif ui_class == 'negate':
            lookup_restaurants()

            if len(informations['suitable_list']) == 0:  # no restaurant found
                return State.RESTAURANT_NOT_FOUND
            else:  # at least one restaurant found
                return State.RESTAURANT_FOUND

        elif ui_class == 'repeat':
            return prev_state

        elif ui_class == 'bye' or ui_class == 'thankyou':
            return State.BYE

        elif ui_class == 'restart':
            for info in informations:
                informations[info] = None
            return State.WELCOME

        return State.AWAIT_COMMAND

    elif current_state == State.PRINT_POSTCODE:
        restaurant = informations['suitable_list'].iloc[0]
        caps_check_print(f"The post code of {restaurant[0]} is {restaurant[6]}.")

        return State.AWAIT_COMMAND

    elif current_state == State.PRINT_ADDRESS:
        restaurant = informations['suitable_list'].iloc[0]
        caps_check_print(f"The address of {restaurant[0]} is {restaurant[5]}.")

        return State.AWAIT_COMMAND

    elif current_state == State.PRINT_PHONENUMBER:
        restaurant = informations['suitable_list'].iloc[0]
        caps_check_print(f"The phone number of {restaurant[0]} is {restaurant[4]}.")

        return State.AWAIT_COMMAND

    elif current_state == State.NO_OTHER_RESTAURANTS:
        caps_check_print("Sorry but there is no other restaurant", end="")
        if informations['area'] != None:
            caps_check_print(f" in the {informations['area']} of town", end="")
        if informations['price'] != None:
            caps_check_print(f" in the {informations['price']} price range", end="")
        if informations['food'] != None:
            caps_check_print(f" serving {informations['food']} food", end="")

        # if there was a requirements but no restaurants met that requirement
        if informations['extra'] != None:
            # check what requirement was asked an build the answer string
            for word in informations['extra'].split():
                if word == 'romantic' or word == 'touristic':
                    caps_check_print(f' that is also {word}', end="")
                if word == 'children':
                    caps_check_print(f' that is also for {word}', end="")
                if word == 'assigned':
                    caps_check_print(f' that also allows for {word} seats', end="")
            informations['extra'] = None
        caps_check_print(".")

        return State.AWAIT_COMMAND

    elif current_state == State.BYE:
        caps_check_print("bye")
        return State.KILL

    return current_state



# User Input
caps_check_print("\nChoose what you want to do:")
caps_check_print("1. Logistic Regression")
caps_check_print("2. Baseline 2")

choice = input("Enter your choice: ")

if choice == "1":
    chosen_model = 1 # logistic regression

elif choice == "2":
    chosen_model = 2 # baseline 2

print("\nDo you want spelling auto-correction?")
print("1. Yes")
print("2. No")

choice = input("Enter your choice: ")

if choice == "1":
    auto_correction = True # logistic regression

elif choice == "2":
    auto_correction = False # baseline 2
    

print("\nDo you want OUTPUT to be in ALL CAPS?")
print("1. YES")
print("2. no")

choice = input("Enter your CHOICE: ")

if choice == "1":
    caps_lock = True # logistic regression

elif choice == "2":
    caps_lock = False # baseline 2
# main loop

if chosen_model == 1:
    model, vocab, LE = train_logistic()

while True:
    new_state = transition(prev_state)
    if new_state == State.KILL:
        caps_check_print("Do you want to start a new conversation?")
        caps_check_print("1. Yes")
        caps_check_print("2. No")
        choice = input("Enter your choice: ")
        if choice == "1":
            informations = {'area': None, 'food': None, 'price': None, 'extra': None, 'suitable_list': None}
            new_state = State.WELCOME
        else:
            break
    prev_state = new_state
