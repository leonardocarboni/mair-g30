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
    5: "RESTAURANT_FOUND",
    6: "RESTAURANT_NOT_FOUND",
    7: "AWAIT_COMMAND",
    8: "GIVE_POSTCODE",
    9: "GIVE_ADRESS",
    10: "GIVE_PHONE_NUMBER",
    11: "NO_ALTERNATIVES",
    
    15: "GOODBYE"
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
    'reqalts': ['alternatives', 'other', 'another', 'different', 'else', 'other'],
    'reqmore': ['more'],
    'request': ['where' '?', 'train', 'taxi', 'plane', 'phone', 'how', 'why', 'number', 'price', 'post', 'code', 'postcode', 'address', 'phonenumber'],
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
        food_filter = (restaurants['food'] == informations['food']).values.tolist()
    if informations['price'] is not None:
        price_filter = (restaurants['pricerange'] == informations['price']).values.tolist()
    if informations['area'] is not None:
        area_filter = (restaurants['area'] == informations['area']).values.tolist()
    
    final_filter = [all(i) for i in zip(food_filter, price_filter, area_filter)]
    
    # print the restaurants that match the user's request on the specified parameters
    informations['suitable_list'] = restaurants[final_filter]

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
            
        if ui_class == 'bye':
            return 15
        
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
        
        return 7
    
    elif old_state == 6:
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
    
    elif old_state == 7:
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
            
        if ui_class == 'request':
            ui_split = user_input.split()
            for i, word in enumerate(ui_split):
                if word == 'postcode' or word == 'post':
                    return 8
                if word == 'address':
                    print('help me 2.0')
                    return 9
                if word == 'phone' or word == 'phonenumber':
                    return 10
        if ui_class == 'reqalts':
            # If there is another restaurant, recommend the different restaurant
            
            # If there is no other restaurant, tell the user
            return 11
        if ui_class == 'bye':
            return 15
        return 7
        
    elif old_state == 8:
        restaurant = informations['suitable_list'].iloc[0]
        print(f"The post code of {restaurant[0]} is {restaurant[6]}.")    
        return 7
    
    elif old_state == 9:
        restaurant = informations['suitable_list'].iloc[0]
        print(f"The address of {restaurant[0]} is {restaurant[5]}.")
        return 7
    
    elif old_state == 10:
        restaurant = informations['suitable_list'].iloc[0]
        print(f"The phone number of {restaurant[0]} is {restaurant[4]}.")
        return 7
    
    elif old_state == 11:
        restaurant = informations['suitable_list'].iloc[0]
        print("Sorry but there is no other restaurant", end="")
        if informations['area'] != None:
            print(f" in the {restaurant[2]} of town", end="")
        if informations['price'] != None:
            print(f" in the {restaurant[1]} price range", end="")
        if informations['food'] != None:
            print(f" serving {restaurant[3]} food", end="")
        print(".")
        return 7
    
    elif old_state == 15:
        print("bye")
        return -1
    return old_state


while True:
    new_state = transition(current_state)
    if new_state == -1:
        break
    current_state = new_state
