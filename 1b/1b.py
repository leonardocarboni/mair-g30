# -*- coding: utf-8 -*-
"""
Part 1b: modelling and implementing a dialog management system
Group G30 (Giacomo Bais, Leonardo Carboni, Merel de Goede, Merel van den Bos)
"""

from cmath import inf
import numpy as np
import pandas as pd

states = {
    1: "WELCOME",
    0: "ASK_INFORMATIONS",
    2: "ASK_AREA",
    3: "ASK_FOOD",
    4: "ASK_PRICERANGE",
    5: "CHECK_RESTAURANTS",
    6: "RESTAURANT_FOUND",
    7: "RESTAURANT_NOT_FOUND",
    8: "AWAIT_COMMAND",
}

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

informations = {'food': None, 'area': None, 'price': None, 'suitable_list': None}

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
    for word in user_input.split():  # split prompt into words
        for key, value in classes.items():  # look for the word in the dictionary
            if word in value:  # if we get a match
                return(key)  # predict the class from the dict
    return(majority)


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


def transition(old_state):
    if old_state == 1:
        print_welcome()
        user_input = input().lower()
        ui_class = extract_class(user_input)
        print("DEBUG - input class: ", ui_class)

        if ui_class == 'inform':
            ui_split = user_input.split()
            extract_params(ui_split)

            print("DEBUG - informations: ", informations)

            if informations['area'] == None:
                return 2
            elif informations['food'] == None:
                return 3
            elif informations['price'] == None:
                return 4
            else:
                return 5
        return 1

    elif old_state == 2:
        print("What area would you like to eat in?")
        user_input = input().lower()
        ui_class = extract_class(user_input)
        print("DEBUG - input class: ", ui_class)

        if ui_class == 'inform':
            ui_split = user_input.split()
            extract_params(ui_split)

            print("DEBUG - informations: ", informations)

            if informations['food'] == None:
                return 3
            elif informations['price'] == None:
                return 4
            else:
                return 5
        return 2
    elif old_state == 3:
        print("What type of food would you like to eat?")
        user_input = input().lower()
        ui_class = extract_class(user_input)
        print("DEBUG - input class: ", ui_class)

        if ui_class == 'inform':
            ui_split = user_input.split()
            extract_params(ui_split)

            print("DEBUG - informations: ", informations)

            if informations['price'] == None:
                return 4
            else:
                return 5
        return 3
    elif old_state == 4:
        print("What price range do you prefer?")
        user_input = input().lower()
        ui_class = extract_class(user_input)
        print("DEBUG - input class: ", ui_class)

        if ui_class == 'inform':
            ui_split = user_input.split()
            extract_params(ui_split)

            print("DEBUG - informations: ", informations)

            if informations['price'] != None:
                return 5
        return 4
    elif old_state == 5:
        # find all restaurants that match the criteria
        informations['suitable_list'] = restaurants[(restaurants['area'] == informations['area']) & (restaurants['food'] == informations['food']) & (restaurants['pricerange'] == informations['price'])].to_numpy()
        if len(informations['suitable_list']) == 0:
                return 7
        else:
            return 6
        
        # if len(restaurants.where((restaurants['area'] == informations['area']) & (restaurants['food'] == informations['food']))) == 1:
        #     suitable_restaurants = [restaurants.where((restaurants['area'] == informations['area']) & (
        #         restaurants['food'] == informations['food']))]
        #     return 6
        # elif len(restaurants.where((restaurants['area'] == informations['area']) & (restaurants['pricerange'] == informations['price']))) == 1:
        #     suitable_restaurants = [restaurants.where((restaurants['area'] == informations['area']) & (
        #         restaurants['pricerange'] == informations['price']))]
        #     return 6
        # elif len(restaurants.where((restaurants['food'] == informations['food']) & (restaurants['pricerange'] == informations['price']))) == 1:
        #     suitable_restaurants = [restaurants.where((restaurants['area'] == informations['area']) & (
        #         restaurants['pricerange'] == informations['price']))]
        #     return 6
        # else:
        #     suitable_restaurants = restaurants.where((restaurants['area'] == informations['area']) and (
        #         restaurants['food'] == informations['food']) and (restaurants['pricerange'] == informations['price']))
        #     if len(suitable_restaurants) == 0:
        #         return 7
        #     else:
        #         print("Here are the restaurants I found:")
        #         print(suitable_restaurants)
        #         return 6
    elif old_state == 6:
        print("Here are the restaurants I found:")
        print(informations['suitable_list'])
        return 8
    return old_state


while True:
    new_state = transition(current_state)
    current_state = new_state
