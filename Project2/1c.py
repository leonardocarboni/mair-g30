# -*- coding: utf-8 -*-
"""
Part 1b: modelling and implementing a dialog management system
Group G30 (Giacomo Bais, Leonardo Carboni, Merel de Goede, Merel van den Bos)
"""

from cmath import inf
import numpy as np
import pandas as pd
from enum import Enum
import utils

# States encoding
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

# Set initial state
prev_state = State.WELCOME


chosen_model = 1 # Decision Tree default

auto_correction = True # use levenshtein edit distance or not

caps_lock = False # Default: not CAPS

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

restaurants = pd.read_csv('Data\\restaurant_info2.csv')

# type of restaurant: ([true inference list], [false inference list])
rules = {
    "romantic": (["long stay"], ["busy"]),
    "children": (["short stay"], ["long stay"]),
    "assigned": (["busy"], ["not busy"]),
    "touristic": (["cheap", "good food"], ["romanian"]),
}


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
        # only corrects distances lower than 2
        for food_type in food_types:
            if min_dist > utils.lev_dist(user_word, food_type) and utils.lev_dist(user_word, food_type) <= 1:
                new_word = food_type
                min_dist = utils.lev_dist(user_word, food_type)
        for price in price_ranges:
            if min_dist > utils.lev_dist(user_word, price) and utils.lev_dist(user_word, price) <= 1:
                new_word = price
                min_dist = utils.lev_dist(user_word, price)
        for area in areas:
            if min_dist > utils.lev_dist(user_word, area) and utils.lev_dist(user_word, area) <= 1:
                new_word = area
                min_dist = utils.lev_dist(user_word, area)
        if new_word is not None:
            return new_word
    return user_word

# Greeting function
def print_welcome():
    utils.caps_check_print("Hello, welcome to the MAIR G30 restaurant system? You can ask for restaurants by area, price range or food type. You can also add additional requirements among: Romantic; Touristic; Children; Assigned seats. How may I help you?", caps_lock)


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
            for key, value in utils.classes.items():  # look for the word in the dictionary
                if word in value:  # if we get a match
                    return(key)  # predict the class from the dict
        return(utils.majority)

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

# Transition function: takes the current state and returns the next depending on user input (classified inside the function)
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
        utils.caps_check_print("What area would you like to eat in?", caps_lock)
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
        utils.caps_check_print("What type of food would you like to eat?", caps_lock)
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
        utils.caps_check_print("What price range do you prefer?", caps_lock)
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
        utils.caps_check_print("Do you have additional requirements?", caps_lock)
        return State.AWAIT_COMMAND

    elif current_state == State.RESTAURANT_FOUND:
        req_string = manage_requirements()

        if len(informations['suitable_list']) == 0:
            return State.RESTAURANT_NOT_FOUND

        restaurant = informations['suitable_list'].iloc[0]
        utils.caps_check_print(f"{restaurant[0]} is a nice place", caps_lock, end="")

        if informations['area'] != None:
            utils.caps_check_print(f" in the {restaurant[2]} of town", caps_lock, end="")
        if informations['price'] != None:
            utils.caps_check_print(f" in the {restaurant[1]} price range", caps_lock, end="")
        if informations['food'] != None:
            utils.caps_check_print(f" serving {restaurant[3]} food", caps_lock, end="")
        utils.caps_check_print(".", caps_lock)

        if req_string != None:
            utils.caps_check_print(req_string, caps_lock)

        return State.AWAIT_COMMAND

    elif current_state == State.RESTAURANT_NOT_FOUND:
        utils.caps_check_print("I'm sorry but there is no restaurant", caps_lock, end="")
        if informations['area'] != None:
            utils.caps_check_print(f" in the {informations['area']} of town", caps_lock, end="")
        if informations['price'] != None:
            utils.caps_check_print(f" in the {informations['price']} price range", caps_lock, end="")
        if informations['food'] != None:
            utils.caps_check_print(f" serving {informations['food']} food", caps_lock, end="")

        # if there was a requirements but no restaurants met that requirement
        if informations['extra'] != None:
            # check what requirement was asked an build the answer string
            for word in informations['extra'].split():
                if word == 'romantic' or word == 'touristic':
                    utils.caps_check_print(f' that is also {word}', caps_lock, end="")
                if word == 'children':
                    utils.caps_check_print(f' that is also for {word}', caps_lock, end="")
                if word == 'assigned':
                    utils.caps_check_print(f' that also allows for {word} seats', caps_lock, end="")
            # if there is no restaurant given the requirements, reset string for inference in case of future suggestions
            informations['extra'] = None
        utils.caps_check_print(".", caps_lock)

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
        utils.caps_check_print(f"The post code of {restaurant[0]} is {restaurant[6]}.", caps_lock)

        return State.AWAIT_COMMAND

    elif current_state == State.PRINT_ADDRESS:
        restaurant = informations['suitable_list'].iloc[0]
        utils.caps_check_print(f"The address of {restaurant[0]} is {restaurant[5]}.", caps_lock)

        return State.AWAIT_COMMAND

    elif current_state == State.PRINT_PHONENUMBER:
        restaurant = informations['suitable_list'].iloc[0]
        utils.caps_check_print(f"The phone number of {restaurant[0]} is {restaurant[4]}.", caps_lock)

        return State.AWAIT_COMMAND

    elif current_state == State.NO_OTHER_RESTAURANTS:
        utils.caps_check_print("Sorry but there is no other restaurant", caps_lock, end="")
        if informations['area'] != None:
            utils.caps_check_print(f" in the {informations['area']} of town", caps_lock, end="")
        if informations['price'] != None:
            utils.caps_check_print(f" in the {informations['price']} price range", caps_lock, end="")
        if informations['food'] != None:
            utils.caps_check_print(f" serving {informations['food']} food", caps_lock, end="")

        # if there was a requirements but no restaurants met that requirement
        if informations['extra'] != None:
            # check what requirement was asked an build the answer string
            for word in informations['extra'].split():
                if word == 'romantic' or word == 'touristic':
                    utils.caps_check_print(f' that is also {word}', caps_lock, end="")
                if word == 'children':
                    utils.caps_check_print(f' that is also for {word}', caps_lock, end="")
                if word == 'assigned':
                    utils.caps_check_print(f' that also allows for {word} seats', caps_lock, end="")
            informations['extra'] = None
        utils.caps_check_print(".", caps_lock)

        return State.AWAIT_COMMAND

    elif current_state == State.BYE:
        utils.caps_check_print("bye", caps_lock)
        return State.KILL

    return current_state



# User Input
utils.caps_check_print("\nChoose what you want to do:", caps_lock)
utils.caps_check_print("1. Decision Tree", caps_lock)
utils.caps_check_print("2. Baseline 2", caps_lock)

choice = input("Enter your choice: ")

if choice == "1":
    chosen_model = 1 # Decision Tree

elif choice == "2":
    chosen_model = 2 # baseline 2

print("\nDo you want spelling auto-correction?")
print("1. Yes")
print("2. No")

choice = input("Enter your choice: ")

if choice == "1":
    auto_correction = True

elif choice == "2":
    auto_correction = False
    

print("\nDo you want OUTPUT to be in ALL CAPS?")
print("1. YES")
print("2. no")

choice = input("Enter your CHOICE: ")

if choice == "1":
    caps_lock = True 

elif choice == "2":
    caps_lock = False

if chosen_model == 1:
    model, LE, vocab = utils.train_tree(utils.df)

# Main loop
while True:
    new_state = transition(prev_state)
    if new_state == State.KILL:
        utils.caps_check_print("Do you want to start a new conversation?", caps_lock)
        utils.caps_check_print("1. Yes", caps_lock)
        utils.caps_check_print("2. No", caps_lock)
        choice = input("Enter your choice: ")
        if choice == "1":
            informations = {'area': None, 'food': None, 'price': None, 'extra': None, 'suitable_list': None}
            new_state = State.WELCOME
        else:
            break
    prev_state = new_state
