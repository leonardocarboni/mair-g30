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

model, LE, vocab = utils.train_MLP(utils.df)


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
    if session['useML']:  # machine learning
        user_data = np.zeros(len(vocab))
        for word in user_input.split():
            word = word.replace('?', '')  # delete question marks from words
            if word in rules.keys():
                return 'inform'
            if session['useAC']:
                word = change_to_lev(word)
            if word in vocab:
                user_data[vocab[word]] += 1
            else:
                user_data[vocab['NEW_WORD']] += 1
        print(LE.inverse_transform(model.predict(user_data.reshape(1, -1))))
        return LE.inverse_transform(model.predict(user_data.reshape(1, -1)))
    else:  # baseline 2
        for word in user_input.split():  # split prompt into words
            word = word.replace('?', '')  # delete question marks from words
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
        word = word.replace('?', '')  # delete question marks from words
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
        elif word == 'food' and prev_word is not None and prev_word not in price_ranges:
            session['informations']['food'] = prev_word
            session.modified = True
        elif word == 'asian':
            session['informations']['food'] = 'asian oriental'
            session.modified = True

        # price ranges
        if word in price_ranges:
            session['informations']['price'] = word
            session.modified = True
        elif word == 'moderately':
            session['informations']['price'] = 'moderate'
            session.modified = True

        # areas
        if word in areas:
            session['informations']['area'] = word
            session.modified = True
        elif word == 'center':
            session['informations']['area'] = 'centre'
            session.modified = True


def manage_requirements():
    """
    It takes the extra information and filters the suitable list based on that information
    :return: a string that explains why the restaurant is suitable for the extra requirement.
    """

    if session['informations']['extra'] == 'romantic':
        temp = pd.DataFrame(session['informations']['suitable_list'])
        temp = temp[temp['stay_length'] == 'long stay']
        session['informations']['suitable_list'] = temp.to_dict()
        session.modified = True
        if len(temp) == 0:
            return ""
        return "The restaurant is romantic because it allows you to stay for a long time."
    elif session['informations']['extra'] == 'children':
        temp = pd.DataFrame(session['informations']['suitable_list'])
        temp = temp[temp['stay_length'] == 'short stay']
        
        if len(temp) == 0:
            return ""
        return "The restaurant is for children because it allows you to stay for a short time."
    elif session['informations']['extra'] == 'assigned':
        temp = pd.DataFrame(session['informations']['suitable_list'])
        temp = temp[temp['crowdedness'] == 'busy']
        session['informations']['suitable_list'] = temp.to_dict()
        session.modified = True
        if len(temp) == 0:
            return ""
        return "The restaurant allows for assigned seats because it is usually busy."
    elif session['informations']['extra'] == 'touristic':
        temp = pd.DataFrame(session['informations']['suitable_list'])
        temp = temp[(temp['pricerange'] == 'cheap') &
                    (temp['food_quality'] == 'good food')]
        session['informations']['suitable_list'] = temp.to_dict()
        session.modified = True
        if len(temp) == 0:
            return ""
        return "The restaurant is touristic because it is cheap and it serves good food."
    else:
        return ""


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
        "Sorry but there is no restaurant", session['useCL'])
    if session['informations']['area'] != None:
        response += utils.caps_check(
            f" in the {session['informations']['area']} of town", session['useCL'])
    if session['informations']['price'] != None:
        response += utils.caps_check(
            f" in the {session['informations']['price']} price range", session['useCL'])
    if session['informations']['food'] != None:
        response += utils.caps_check(
            f" serving {session['informations']['food']} food", session['useCL'])
        
    if session['informations']['extra'] != None:
        # check what requirement was asked an build the answer string
        for word in session['informations']['extra'].split():
            word = word.replace('?', '') # delete question marks from words
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
        session.modified = True
    response += utils.caps_check(".", session['useCL'])    
    return response

def get_restaurant_found():
    restaurant = pd.DataFrame(session['informations']['suitable_list']).iloc[0]
    response = utils.caps_check(
        f"{restaurant['restaurantname']} is a nice place", session['useCL'])
    if session['informations']['area'] != None:
        response += utils.caps_check(
            f" in the {restaurant['area']} of town", session['useCL'])
    if session['informations']['price'] != None:
        response += utils.caps_check(
            f" in the {restaurant['pricerange']} price range", session['useCL'])
    if session['informations']['food'] != None:
        response += utils.caps_check(
            f" serving {restaurant['food']} food", session['useCL'])
    response += "."

    if session['informations']['extra'] != None:
        req = manage_requirements()
        if req != '':
            response += utils.caps_check(" " +
                                     req, session['useCL'])
        else:
            response = get_no_restaurant_found()
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
            word = word.replace('?', '') # delete question marks from words
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
        session.modified = True
        
    return response


def get_postcode():
    restaurant = pd.DataFrame(session['informations']['suitable_list']).iloc[0]
    return utils.caps_check(f"The post code of {restaurant['restaurantname']} is {restaurant['postcode']}.", session['useCL'])


def get_address():
    restaurant = pd.DataFrame(session['informations']['suitable_list']).iloc[0]
    return utils.caps_check(f"The address of {restaurant['restaurantname']} is {restaurant['addr']}.", session['useCL'])


def get_phonenumber():
    restaurant = pd.DataFrame(session['informations']['suitable_list']).iloc[0]
    return utils.caps_check(f"The phone number of {restaurant['restaurantname']} is {restaurant['phone']}.", session['useCL'])


def get_bye():
    response = "Goodbye!"
    if session['informations']['attempt'] >= 11:
        response += " Please leave complete the survey on <a href='https://forms.office.com/Pages/DesignPageV2.aspx?subpage=design&FormId=oFgn10akD06gqkv5WkoQ5z6fwTBqk1NEjDs1bydB55RUNlpQVEc2RzRKRFZIM0lLMERMWTNIM0g0TS4u&Token=7c62bf5a858e4276aace5efa9ccda3ae'>this link</a>."
    return utils.caps_check(response, session['useCL'])


def get_response(user_message):

    user_input = user_message.lower()
    if user_input == "/state":
        return "+ STATE: " + session['state']
    elif user_input == "/debug":
        return f"+ DEBUG: {str(session['informations'])}, {str(session['useML']), str(session['useCL']), str(session['useAC'])}"
    elif user_input == "/df":
        return str(pd.DataFrame(session['informations']['suitable_list']))
    ui_class = extract_class(user_input)

    utils.save_msg_on_file(user_input, False)

    if ui_class == 'bye':
        session['state'] = 13
        session.modified = True
        return get_bye()

    elif ui_class == 'restart':
        for info in session['informations']:
            if info != 'attempt':
                session['informations'][info] = None
        session['informations']['attempt'] += 1
        session['state'] = 1
        session.modified = True
        if session['informations']['attempt'] >= 5:
            return " Please leave complete the survey on <a href='https://forms.office.com/Pages/DesignPageV2.aspx?subpage=design&FormId=oFgn10akD06gqkv5WkoQ5z6fwTBqk1NEjDs1bydB55RUNlpQVEc2RzRKRFZIM0lLMERMWTNIM0g0TS4u&Token=7c62bf5a858e4276aace5efa9ccda3ae'>this link</a>."
        return "RESTARTED: " + get_welcome()

    if session['state'] == 1:
        if ui_class == 'hello':
            return get_welcome()
        if ui_class == 'inform' or ui_class == 'deny':
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()

            if len(session['informations']['suitable_list']['addr']) == 0:
                session['state'] = 8
                session.modified = True
                return get_no_restaurant_found()
            if len(session['informations']['suitable_list']['addr']) == 1:
                session['state'] = 8
                session.modified = True
                return get_restaurant_found()

            if session['informations']['area'] == None:
                session['state'] = 2
                session.modified = True
                return get_ask_area()
            elif session['informations']['food'] == None:
                session['state'] = 3
                session.modified = True
                return get_ask_food()
            elif session['informations']['price'] == None:
                session['state'] = 4
                session.modified = True
                return get_ask_price()
            else:
                session['state'] = 8
                session.modified = True
                return get_ask_requirements()
        return "Please provide a food type, area or price range for the restaurant."

    elif session['state'] == 2:
        if ui_class == 'inform' or ui_class == 'deny':
            print(session['informations'])
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()

            if len(session['informations']['suitable_list']['addr']) == 0:
                session['state'] = 8
                session.modified = True
                return get_no_restaurant_found()
            if len(session['informations']['suitable_list']['addr']) == 1:
                session['state'] = 8
                session.modified = True
                return get_restaurant_found()

            elif session['informations']['food'] == None:
                session['state'] = 3
                session.modified = True
                return get_ask_food()
            elif session['informations']['price'] == None:
                session['state'] = 4
                session.modified = True
                return get_ask_price()
            else:
                session['state'] = 8
                session.modified = True
                return get_ask_requirements()
        return "Please provide a food type, area or price range for the restaurant."

    elif session['state'] == 3:
        if ui_class == 'inform' or ui_class == 'deny':
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()

            if len(session['informations']['suitable_list']['addr']) == 0:
                session['state'] = 8
                session.modified = True
                return get_no_restaurant_found()
            if len(session['informations']['suitable_list']['addr']) == 1:
                session['state'] = 8
                session.modified = True
                return get_restaurant_found()

            if session['informations']['price'] == None:
                session['state'] = 4
                session.modified = True
                return get_ask_price()
            else:
                session['state'] = 8
                session.modified = True
                return get_ask_requirements()
        return "Please provide a food type, area or price range for the restaurant."

    elif session['state'] == 4:
        if ui_class == 'inform' or ui_class == 'deny':
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()

            if len(session['informations']['suitable_list']['addr']) == 0:
                session['state'] = 8
                session.modified = True
                return get_no_restaurant_found()
            if len(session['informations']['suitable_list']['addr']) == 1:
                session['state'] = 8
                session.modified = True
                return get_restaurant_found()
            else:
                session['state'] = 8
                session.modified = True
                return get_ask_requirements()

    elif session['state'] == 8:
        if ui_class == 'inform':
            ui_split = user_input.split()
            extract_params(ui_split)
            lookup_restaurants()

            if len(session['informations']['suitable_list']['addr']) == 0:
                session['state'] = 8
                session.modified = True
                return get_no_restaurant_found()
            if len(session['informations']['suitable_list']['addr']) == 1:
                session['state'] = 8
                session.modified = True
                return get_restaurant_found()

            for word in ui_split:
                word = word.replace('?', '') # delete question marks from words
                if word in rules.keys():
                    session['informations']['extra'] = word
                    session.modified = True

            # if a requirement was given suggest the restaurant
            if session['informations']['extra'] != None:
                session['state'] = 8
                session.modified = True
                return get_restaurant_found()
            # if a requirement wasn't given, user is trying to change the main 3 infos and we go back
            # these 3 checks are probably useless #
            elif session['informations']['area'] == None:  # area not specified -> ask area
                session['state'] = 2
                session.modified = True
                return get_ask_area()
            # # food type not specified -> ask food type
            elif session['informations']['food'] == None:
                session['state'] = 3
                session.modified = True
                return get_ask_food()
            # # price range not specified -> ask price range
            elif session['informations']['price'] == None:
                session['state'] = 4
                session.modified = True
                return get_ask_price()
            else:  # if we have all informations but a preference wasn't given, then we ask for them
                return get_ask_requirements()

        elif ui_class == 'request':
            ui_split = user_input.split()

            for word in ui_split:
                word = word.replace('?', '') # delete question marks from words
                if word == 'postcode' or word == 'post':
                    return get_postcode()
                elif word == 'address':
                    return get_address()
                elif word == 'phone' or word == 'phonenumber':
                    return get_phonenumber()

        elif ui_class == 'reqalts':
            # If there is another restaurant, recommend the different restaurant
            if len(session['informations']['suitable_list']['addr']) > 1:
                session['informations']['suitable_list'] = (pd.DataFrame(
                    session['informations']['suitable_list'])[1:]).to_dict()
                session['state'] = 8
                session.modified = True
                return get_restaurant_found()
            # If there is no other restaurant, tell the user
            session['state'] = 8
            session.modified = True
            return get_no_other_restaurants()

        # elif ui_class == 'negate':
        #     lookup_restaurants()

        #     if len(session['informations']['suitable_list']['addr']) == 0:  # no restaurant found
        #         return State.RESTAURANT_NOT_FOUND
        #     else:  # at least one restaurant found
        #         return State.RESTAURANT_FOUND
    return "Please try again rephrasing the sentence."
