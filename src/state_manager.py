from joblib import load
import pandas as pd
import random
import re
from src.classifier import Classifier
from src.preference_extractor import extract_preferences_from_sentence, extract_post_or_phone
from Levenshtein import distance

restaurants = pd.read_csv(r'../data/restaurant_info.csv')


class State:
    def __init__(self, food_type, price, area):
        self.food_type = food_type
        self.price = price
        self.area = area
        self.suggestions = None
        self.current_suggestion = 0
        self.state_number = 1

    def get_question(self):
        if self.state_number == 1:
            return "Hello, how may I help you today? "
        elif self.state_number == 2:
            return "What kind of food should the restaurant serve? "
        elif self.state_number == 3:
            return "How expensive should the restaurant be? "
        elif self.state_number == 4:
            return "Where should the restaurant be located? "
        elif self.state_number == 5:
            suggestion = self.get_suggestion()
            return self.generate_random_text_suggestion(
                suggestion) + "\nDo you want any information about it, such as the phone or post code? "
        elif self.state_number == 6:
            post_code = self.suggestions.iloc[self.current_suggestion].postcode
            if post_code is None:
                return "I'm sorry, i don't have the postcode for this restaurant. Any other information request? "
            return f"It is located at {post_code}. Any other information request? "
        elif self.state_number == 7:
            phone_number = self.suggestions.iloc[self.current_suggestion].phone
            if phone_number is None:
                return "I'm sorry, i don't have the phone number for this restaurant. Any other information request? "
            return f"The phone number is {phone_number}. Any other information request? "
        else:
            return "Goodbye"

    @staticmethod
    def generate_random_text_suggestion(suggestion):
        responses = [f"I would suggest {suggestion.restaurantname} that serves {suggestion.food} food at "
                     f"{suggestion.pricerange} price. The address is {suggestion.addr}",
                     f"{suggestion.restaurantname} is a nice restaurant in {suggestion.addr} that has "
                     f"{suggestion.food}, and has {suggestion.pricerange} price",
                     f"Located in {suggestion.addr} and with a {suggestion.pricerange} price, "
                     f"{suggestion.restaurantname} is a good choice"]
        return random.choice(responses)

    def generate_suggestions(self):
        self.suggestions = restaurants.copy()  # at the start all restaurants are matches

        if self.food_type is not None:  # check if the user has specified a food type
            if self.food_type in self.suggestions['food']:  # check if there is an exact match
                self.suggestions = self.suggestions[
                    self.suggestions['food'].str.contains(self.food_type)]  # filter to only that food type
            else:  # if not we want to iterate all of the food types and compute the Levenshtein distance
                for i in self.suggestions.loc[:, 'food']:
                    if distance(self.food_type, i) >= 2:
                        # >= 2 is an arbitrary cut-off point, we could do sth fancy instead
                        self.suggestions = self.suggestions[self.suggestions['food'] != i]

        # the following two are the same as what happened above but w/ the other features
        if self.price is not None:
            if self.price in self.suggestions['pricerange']:
                self.suggestions = self.suggestions[self.suggestions['pricerange'].str.contains(self.price)]
            else:
                for i in self.suggestions.loc[:, 'pricerange']:
                    if distance(self.price, i) >= 2:
                        self.suggestions = self.suggestions[self.suggestions['pricerange'] != i]

        if self.area is not None:
            if self.area in self.suggestions['area']:
                self.suggestions = self.suggestions[self.suggestions['area'].str.contains(self.area)]
            else:
                for i in self.suggestions.loc[:, 'area']:
                    if distance(self.area, i) >= 2:
                        self.suggestions = self.suggestions[self.suggestions['area'] != i]

        return self.suggestions

    def get_suggestion(self):
        if self.suggestions is None:
            self.generate_suggestions()

        # Create suggestion
        suggestion = self.suggestions.iloc[self.current_suggestion]

        # Update current pointer to 1 higher, or the first suggestion if the suggestions are exhausted
        self.current_suggestion = (self.current_suggestion + 1) % len(self.suggestions)

        return suggestion


def initialize_state(classifier: Classifier):
    initial_sentence = input("Hello, how may I help you today? ")

    state = State(food_type=None, price=None, area=None)
    update_state(state, classifier, initial_sentence)

    return state


def update_state(state: State, classifier: Classifier, sentence: str):
    # Ask for more info if it is required
    if state.food_type is None or state.price is None or state.area is None:
        new_food_type, new_area, new_price = extract_from_sentence(sentence)
        if new_food_type is not None:
            state.food_type = new_food_type
        if new_price is not None:
            state.price = new_price
        if new_area is not None:
            state.area = new_area

        if state.food_type is None:
            state.state_number = 2
        elif state.price is None:
            state.state_number = 3
        elif state.area is None:
            state.state_number = 4
        else:
            state.state_number = 5

    # Otherwise, check if a request needs to be made
    else:
        bow = load("../models/bow.joblib")
        clf = classifier.load_from_file()
        response_type = clf.transform_and_predict(sentence, bow)
        # If an alternative is requested, no state update is needed
        if state.state_number == 5 and response_type == "reqalts":
            return

        # If a request is made, the state should be updated to give details in the next iteration
        # if response_type == "request":
        request_type = determine_post_or_phone_question(sentence)
        if request_type == "post":
            state.state_number = 6
        elif request_type == "phone":
            state.state_number = 7
        else:
            state.state_number = 8


def extract_from_sentence(sentence):
    return extract_preferences_from_sentence(sentence)


def determine_post_or_phone_question(sentence):
    return extract_post_or_phone(sentence)
