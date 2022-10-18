from cmath import inf
import re
import numpy as np
import pandas as pd
from enum import Enum
import utils

from flask import session

# States encoding


class State(Enum):
    NOT_STARTED = 0
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

restaurants = pd.read_csv('data/restaurant_info2.csv')

# type of restaurant: ([true inference list], [false inference list])
rules = {
    "romantic": (["long stay"], ["busy"]),
    "children": (["short stay"], ["long stay"]),
    "assigned": (["busy"], ["not busy"]),
    "touristic": (["cheap", "good food"], ["romanian"]),
}

model, LE, vocab = utils.train_tree(utils.df)


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


def extract_class(user_input):
    """
    It takes in a string, and returns the class that the model predicts

    :param user_input: The user's input
    :return: The class of the input
    """
    if session['useDT']:  # decision Tree
        user_data = np.zeros(len(vocab))
        for word in user_input.split():
            if session['useAC']:
                word = change_to_lev(word)
            if word in vocab:
                user_data[vocab[word]] += 1
            else:
                user_data[vocab['NEW_WORD']] += 1
        return LE.inverse_transform(model.predict(user_data.reshape(1, -1)))
    else:  # baseline 2
        for word in user_input.split():  # split prompt into words
            if session['useAC']:
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
        if session['useAC']:
            word = change_to_lev(word)
            if i != 0:
                prev_word = change_to_lev(ui_split[i-1])
        elif i != 0:
            prev_word = ui_split[i-1]
        # type of food
        if word in food_types:
            session['informations']['food'] = word
            session.modified = True
            print(session['informations']['food'])
        elif word == 'food' and prev_word is not None and prev_word not in price_ranges:
            session['informations']['food'] = prev_word
        elif word == 'asian':
            session['informations']['food'] = 'asian oriental'

        # price ranges
        if word in price_ranges:
            session['informations']['price'] = word
        elif word == 'moderately':
            session['informations']['price'] = 'moderate'

        # areas
        if word in areas:
            session['informations']['area'] = word
        elif word == 'center':
            session['informations']['area'] = 'centre'


def manage_requirements():
    """
    It takes the extra information and filters the suitable list based on that information
    :return: a string that explains why the restaurant is suitable for the extra requirement.
    """
    
    #TODO: FIX
    if session['informations']['extra'] == 'romantic':
        session['informations']['suitable_list'] = session['informations']['suitable_list'][session['informations']['suitable_list']
                                                                                            ['stay_length'] == 'long stay']
        return "The restaurant is romantic because it allows you to stay for a long time."
    if session['informations']['extra'] == 'children':
        session['informations']['suitable_list'] = session['informations']['suitable_list'][session['informations']['suitable_list']
                                                                                            ['stay_length'] == 'short stay']
        return "The restaurant is for children because it allows you to stay for a short time."
    if session['informations']['extra'] == 'assigned':
        session['informations']['suitable_list'] = session['informations']['suitable_list'][session['informations']['suitable_list']
                                                                                            ['crowdedness'] == 'busy']
        return "The restaurant allows for assigned seats because it is usually busy."
    if session['informations']['extra'] == 'touristic':
        session['informations']['suitable_list'] = session['informations']['suitable_list'][(session['informations']['suitable_list']['pricerange'] == 'cheap') & (
            session['informations']['suitable_list']['food_quality'] == 'good food')]
        return "The restaurant is touristic because it is cheap and it serves good food."


def lookup_restaurants():
    """
    It filters the restaurants dataframe based on the user's input and stores the filtered dataframe in
    the informations dictionary
    """
    food_filter = [True] * len(restaurants)
    price_filter = [True] * len(restaurants)
    area_filter = [True] * len(restaurants)

    if session['informations']['food'] is not None:
        food_filter = (restaurants['food'] ==
                       session['informations']['food']).values.tolist()
    if session['informations']['price'] is not None:
        price_filter = (restaurants['pricerange'] ==
                        session['informations']['price']).values.tolist()
    if session['informations']['area'] is not None:
        area_filter = (restaurants['area'] ==
                       session['informations']['area']).values.tolist()

    final_filter = [all(i)
                    for i in zip(food_filter, price_filter, area_filter)]

    # print the restaurants that match the user's request on the specified parameters
    session['informations']['suitable_list'] = restaurants[final_filter].to_dict()


def get_welcome():
    return utils.caps_check("Hello, welcome to the MAIR G30 restaurant system? You can ask for restaurants by area, price range or food type. You can also add additional requirements among: Romantic; Touristic; Children; Assigned seats. How may I help you?", session['useCL'])


def get_no_restaurant_found():
    response = utils.caps_check(
        "Sorry but there is no other restaurant", session['useCL'])
    if session['informations']['area'] != None:
        response += utils.caps_check(
            f" in the {session['informations']['area']} of town", session['useCL'])
    if session['informations']['price'] != None:
        response += utils.caps_check(
            f" in the {session['informations']['price']} price range", session['useCL'])
    if session['informations']['food'] != None:
        response += utils.caps_check(
            f" serving {session['informations']['food']} food", session['useCL'])
    response += utils.caps_check(".", session['useCL'])
    return response


def get_restaurant_found():
    restaurant = session['informations']['suitable_list'][0]
    response = utils.caps_check(
        f"{restaurant[0]} is a nice place", session['useCL'])
    if session['informations']['area'] != None:
        response += utils.caps_check(
            f" in the {restaurant[2]} of town", session['useCL'])
    if session['informations']['price'] != None:
        response += utils.caps_check(
            f" in the {restaurant[1]} price range", session['useCL'])
    if session['informations']['food'] != None:
        response += utils.caps_check(
            f" serving {restaurant[3]} food", session['useCL'])
    response += "."

    if session['informations']['extra'] != None:
        response += utils.caps_check(" " +
                                     manage_requirements(), session['useCL'])
    return response


def get_ask_area():
    return utils.caps_check("What area of town do you want to eat in?", session['useCL'])


def get_ask_price():
    return utils.caps_check("What price range do you want to eat in?", session['useCL'])


def get_ask_food():
    return utils.caps_check("What type of food would you like to eat?", session['useCL'])


def get_ask_requirements():
    return utils.caps_check("What other requirements do you have?", session['useCL'])


def get_no_other_restaurants():
    response = "Sorry but there is no other restaurant"
    if session['informations']['area'] != None:
        response += utils.caps_check(
            f" in the {session['informations']['area']} of town", session['useCL'])
    if session['informations']['price'] != None:
        response += utils.caps_check(
            f" in the {session['informations']['price']} price range", session['useCL'])
    if session['informations']['food'] != None:
        response += utils.caps_check(
            f" serving {session['informations']['food']} food", session['useCL'])

    # if there was a requirements but no restaurants met that requirement
    if session['informations']['extra'] != None:
        # check what requirement was asked an build the answer string
        for word in session['informations']['extra'].split():
            if word == 'romantic' or word == 'touristic':
                response += utils.caps_check(
                    f' that is also {word}', session['useCL'])
            if word == 'children':
                response += utils.caps_check(
                    f' that is also for {word}', session['useCL'])
            if word == 'assigned':
                response += utils.caps_check(
                    f' that also allows for {word} seats', session['useCL'])
        session['informations']['extra'] = None


def get_postcode():
    restaurant = session['informations']['suitable_list'][0]
    return utils.caps_check(f"The post code of {restaurant[0]} is {restaurant[6]}.", session['useCL'])


def get_address():
    restaurant = session['informations']['suitable_list'][0]
    return utils.caps_check(f"The address of {restaurant[0]} is {restaurant[5]}.", session['useCL'])


def get_phonenumber():
    restaurant = session['informations']['suitable_list'][0]
    return utils.caps_check(f"The phone number of {restaurant[0]} is {restaurant[4]}.", session['useCL'])


def get_bye():
    return utils.caps_check("Goodbye!", session['useCL'])


def get_response(user_message):
    global prev_state
    
    user_input = user_message.lower()
    if user_input == "/state":
        return "+ STATE: " + prev_state.name
    elif user_input == "/debug":
        return f"+ DEBUG: {str(session['informations'])}, {str(session['useDT']), str(session['useCL']), str(session['useAC'])}"
    ui_class = extract_class(user_input)
    
    if ui_class == 'bye':
        prev_state = State.BYE
        return get_bye()

    elif ui_class == 'restart':
        for info in session['informations']:
            session['informations'][info] = None
        prev_state = State.WELCOME
        return "RESTARTED: " + get_welcome()

    if prev_state == State.WELCOME:
        if ui_class == 'hello':
            return get_welcome()
        if ui_class == 'inform' or ui_class == 'deny':
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()
            print(session['informations'])

            if len(session['informations']['suitable_list']) == 0:
                prev_state = State.AWAIT_COMMAND
                return get_no_restaurant_found()
            if len(session['informations']['suitable_list']) == 1:
                prev_state = State.AWAIT_COMMAND
                return get_restaurant_found()

            if session['informations']['area'] == None:
                prev_state = State.ASK_AREA
                return get_ask_area()
            elif session['informations']['food'] == None:
                prev_state = State.ASK_FOOD
                return get_ask_food()
            elif session['informations']['price'] == None:
                prev_state = State.ASK_PRICE
                return get_ask_price()
            else:
                prev_state = State.AWAIT_COMMAND
                return get_ask_requirements()
        return "aaaaaaaaaaaaa"

    elif prev_state == State.ASK_AREA:
        if ui_class == 'inform' or ui_class == 'deny':
            print(session['informations'])
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()

            if len(session['informations']['suitable_list']) == 0:
                prev_state = State.AWAIT_COMMAND
                return get_no_restaurant_found()
            if len(session['informations']['suitable_list']) == 1:
                prev_state = State.AWAIT_COMMAND
                return get_restaurant_found()

            elif session['informations']['food'] == None:
                prev_state = State.ASK_FOOD
                return get_ask_food()
            elif session['informations']['price'] == None:
                prev_state = State.ASK_PRICE
                return get_ask_price()
            else:
                prev_state = State.AWAIT_COMMAND
                return get_ask_requirements()
        return "bbbbbbbbbbbbbb"

    elif prev_state == State.ASK_FOOD:
        if ui_class == 'inform' or ui_class == 'deny':
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()

            if len(session['informations']['suitable_list']) == 0:
                prev_state = State.AWAIT_COMMAND
                return get_no_restaurant_found()
            if len(session['informations']['suitable_list']) == 1:
                prev_state = State.AWAIT_COMMAND
                return get_restaurant_found()

            if session['informations']['price'] == None:
                prev_state = State.ASK_PRICE
                return get_ask_price()
            else:
                prev_state = State.AWAIT_COMMAND
                return get_ask_requirements()
        return "cccccccccccccc"

    elif prev_state == State.ASK_PRICE:
        if ui_class == 'inform' or ui_class == 'deny':
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()

            if len(session['informations']['suitable_list']) == 0:
                prev_state = State.AWAIT_COMMAND
                return get_no_restaurant_found()
            if len(session['informations']['suitable_list']) == 1:
                prev_state = State.AWAIT_COMMAND
                return get_restaurant_found()
            else:
                prev_state = State.AWAIT_COMMAND
                return get_ask_requirements()

    elif prev_state == State.AWAIT_COMMAND:
        if ui_class == 'inform':
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()

            if len(session['informations']['suitable_list']) == 0:
                prev_state = State.AWAIT_COMMAND
                return get_no_restaurant_found()
            if len(session['informations']['suitable_list']) == 1:
                prev_state = State.AWAIT_COMMAND
                return get_restaurant_found()

            for word in ui_split:
                if word in rules.keys():
                    session['informations']['extra'] = word

            # if a requirement was given suggest the restaurant
            if session['informations']['extra'] != None:
                prev_state = State.AWAIT_COMMAND
                return get_restaurant_found()
            # if a requirement wasn't given, user is trying to change the main 3 infos and we go back
            # these 3 checks are probably useless #
            elif session['informations']['area'] == None:  # area not specified -> ask area
                prev_state = State.ASK_AREA
                return get_ask_area()
            # # food type not specified -> ask food type
            elif session['informations']['food'] == None:
                prev_state = State.ASK_FOOD
                return get_ask_food()
            # # price range not specified -> ask price range
            elif session['informations']['price'] == None:
                prev_state = State.ASK_PRICE
                return get_ask_price()
            else:  # if we have all informations but a preference wasn't given, then we ask for them
                return get_ask_requirements()

        elif ui_class == 'request':
            ui_split = user_input.split()

            for word in ui_split:
                if word == 'postcode' or word == 'post':
                    return get_postcode()
                elif word == 'address':
                    return get_address()
                elif word == 'phone' or word == 'phonenumber':
                    return get_phonenumber()

        elif ui_class == 'reqalts':
            # If there is another restaurant, recommend the different restaurant
            if len(session['informations']['suitable_list']) > 1:
                session['informations']['suitable_list'] = session['informations']['suitable_list'][1:]
                prev_state = State.AWAIT_COMMAND
                return get_restaurant_found()
            # If there is no other restaurant, tell the user
            prev_state = State.NO_OTHER_RESTAURANTS
            return get_no_other_restaurants()

        # reqmore not implemented bc it doesn't make sense

        elif ui_class == 'negate':
            lookup_restaurants()

            if len(session['informations']['suitable_list']) == 0:  # no restaurant found
                return State.RESTAURANT_NOT_FOUND
            else:  # at least one restaurant found
                return State.RESTAURANT_FOUND
    return "not implemented"
