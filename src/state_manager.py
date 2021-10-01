from Levenshtein import distance
from joblib import load
from tts import TTS
import pandas as pd
import random
from src.ml.classifiers.classifier import Classifier
from src.preference_extractor import extract_preferences_from_sentence, extract_post_or_phone

restaurants = pd.read_csv(r'../data/restaurant_info.csv')


class State:
    def __init__(self, food_type, price, area, settings):
        self.food_type = food_type
        self.price = price
        self.area = area
        self.suggestions = None
        self.current_suggestion = 0
        self.state_number = 1
        self.settings = settings
        self.state_responses = {
            "1": "Hello, how may I help you today? ",
            "2": "What kind of food should the restaurant serve? ",
            "3": "Should the restaurant be expensive, moderate or cheap? ",
            "4": "Where should the restaurant be located? ",
            "5": "\nDo you want any information about it, such as the phone or post code? ",
            "6_no_postcode":
                "I'm sorry, i don't have the postcode for this restaurant. Any other information request? ",
            "6_give_postcode": lambda post_code: f"It is located at {post_code}. Any other information request? ",
            "7_no_phone":
                "I'm sorry, i don't have the phone number for this restaurant. Any other information request? ",
            "7_give_phone": lambda phone_number: f"The phone number is {phone_number}. Any other information request? ",
            "goodbye": "Goodbye"
        } if not self.settings["informal"] else \
            {
                "1": "What's up, need anything? ",
                "2": "What kind of food do you want? ",
                "3": "How much money do you have? Do you want a cheap restaurant, or a moderate or expensive one? ",
                "4": "Where do you want the place to be at?",
                "5": "\nWant to hear about it? I might have the phone number or post code. ",
                "6_no_postcode": "Oh my bad, I don't have the postcode. Need anything else? ",
                "6_give_postcode": lambda post_code: f"Go to {post_code}. Need anything else? ",
                "7_no_phone":
                    "Oh my bad, I don't have the phone number. Need anything else? ",
                "7_give_phone": lambda phone_number: f"Just call {phone_number}. Need anything else? ",
                "goodbye": "See ya"
            }

    def get_question(self):
        if self.state_number == 1:
            return self.state_responses["1"]
        elif self.state_number == 2:
            return self.state_responses["2"]
        elif self.state_number == 3:
            return self.state_responses["3"]
        elif self.state_number == 4:
            return self.state_responses["4"]
        elif self.state_number == 5:
            suggestion = self.get_suggestion()
            if suggestion is not None:
                return self.generate_random_text_suggestion(
                    suggestion) + self.state_responses["5"]
            else:
                return self.generate_random_text_suggestion_negative()
        elif self.state_number == 6:
            post_code = self.suggestions.iloc[self.current_suggestion].postcode
            if post_code is None:
                return self.state_responses["6_no_postcode"]
            return self.state_responses["6_give_postcode"](post_code)
        elif self.state_number == 7:
            phone_number = self.suggestions.iloc[self.current_suggestion].phone
            if phone_number is None:
                return self.state_responses["7_no_phone"]
            return self.state_responses["7_give_phone"](phone_number)
        else:
            return self.state_responses["goodbye"]

    def generate_random_text_suggestion(self, suggestion):
        responses = [f"I would suggest {suggestion.restaurantname} that serves {suggestion.food} food at "
                     f"{suggestion.pricerange} price. The address is {suggestion.addr}",
                     f"{suggestion.restaurantname} is a nice restaurant in {suggestion.addr} that has "
                     f"{suggestion.food}, and has {suggestion.pricerange} price",
                     f"Located in {suggestion.addr} and with a {suggestion.pricerange} price, "
                     f"{suggestion.restaurantname} is a good choice"] if not self.settings["informal"] else \
            [f"You might want to check out {suggestion.restaurantname}, they got {suggestion.food} food at "
             f"{suggestion.pricerange} price. You can find it at {suggestion.addr}",
             f"{suggestion.restaurantname} is a awesome, it's in {suggestion.addr} and has "
             f"{suggestion.food}, for a {suggestion.pricerange} price",
             f"There's this place near {suggestion.addr}, with {suggestion.pricerange} price, "
             f"it's called {suggestion.restaurantname}, pretty good stuff in my opinion."]
        return random.choice(responses)

    def generate_random_text_suggestion_negative(self):
        responses = [f"No restaurants meet the specified requirements. Please give different specifications",
                     f"I'm sorry, no fitting restaurant has been found. Please give different specifications",
                     f"A restaurant with such requirements does not exist, please give different specifications"] \
            if not self.settings["informal"] else \
            [f"You need to try something else, I don't know anything like that.",
             f"Nope, sorry, can't seem to find anything like that.",
             f"Maybe there's something like that, but I definitively don't know it."]
        return random.choice(responses)

    def generate_suggestions(self):
        self.suggestions = restaurants.copy()  # at the start all restaurants are matches

        if self.food_type is not None:  # check if the user has specified a food type
            if self.food_type in self.suggestions['food']:  # check if there is an exact match
                self.suggestions = self.suggestions[
                    self.suggestions['food'].str.contains(self.food_type)]  # filter to only that food type
            elif self.food_type == "dontcare":
                # TODO
                pass

        # the following two are the same as what happened above but w/ the other features
        if self.price is not None:
            if self.price in self.suggestions['pricerange']:
                self.suggestions = self.suggestions[self.suggestions['pricerange'].str.contains(self.price)]
            elif self.price == "dontcare":
                # TODO
                pass

        if self.area is not None:
            if self.area in self.suggestions['area']:
                self.suggestions = self.suggestions[self.suggestions['area'].str.contains(self.area)]
            elif self.area == "dontcare":
                # TODO
                pass

        return self.suggestions

    def get_suggestion(self):
        if self.suggestions is None:
            self.generate_suggestions()
        if self.suggestions.empty:
            return None
        # Create suggestion
        suggestion = self.suggestions.iloc[self.current_suggestion]

        # Update current pointer to 1 higher, or the first suggestion if the suggestions are exhausted
        self.current_suggestion = (self.current_suggestion + 1) % len(self.suggestions)

        return suggestion


def initialize_state(classifier: Classifier, settings: dict):
    welcome_message = "Hello, how may I help you today? " if not settings['informal'] else \
        "Hey how are you doing, need some help? "
    if settings['useCaps']:
        welcome_message = welcome_message.upper()
    if settings["tts"]:
        tts = TTS()
        tts.speak(welcome_message)
    initial_sentence = input(welcome_message)

    state = State(food_type=None, price=None, area=None, settings=settings)
    update_state(state, classifier, initial_sentence)

    return state


def update_state(state: State, classifier: Classifier, sentence: str):
    # Ask for more info if it is required
    if state.food_type is None or state.price is None or state.area is None:
        new_food_type, new_area, new_price = extract_from_sentence(sentence, state)
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

    # If the state can't generate a suggestion, try again
    elif state.state_number == 5 and len(state.suggestions) == 0:
        state.area = None
        state.food_type = None
        state.price = None
        state.state_number = 2
        return

    # Otherwise, check if a request needs to be made
    else:
        tfidf = load("../models/tfidf.joblib")
        clf = classifier.load_from_file()
        response_type = clf.transform_and_predict(sentence, tfidf)
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


def extract_from_sentence(sentence, state):
    return extract_preferences_from_sentence(sentence, state.settings["levenshteinDistance"], state.state_number)


def determine_post_or_phone_question(sentence):
    return extract_post_or_phone(sentence)
