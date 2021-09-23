from joblib import load
import pandas as pd
from src.classifier import Classifier
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
        if self.state_number is 1:
            return "Hello, how may I help you today?"
        elif self.state_number is 2:
            return "What kind of food should the restaurant serve?"
        elif self.state_number is 3:
            return "How expensive should the restaurant be?"
        elif self.state_number is 4:
            return "Where should the restaurant be located?"
        elif self.state_number is 5:
            return f"I would suggest {self.get_suggestion().name}"
        elif self.state_number is 6:
            return f"It is located at {self.suggestions[self.current_suggestion].post_code}"
        elif self.state_number is 7:
            return f"The phone number is {self.suggestions[self.current_suggestion].phone_number}"
        else:
            return "Goodbye"

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

        # !! please someone make sure we don't need these
        # remove the features we don't need
        self.suggestions.drop(labels=['pricerange', 'area', 'food', 'addr'], axis=1, inplace=True)
        self.suggestions = self.suggestions.to_numpy()  # convert from dataframe to an array
        return self.suggestions

    def get_suggestion(self):
        if self.suggestions is None:
            self.generate_suggestions()

        # Create suggestion
        suggestion = self.suggestions(self.current_suggestion)

        # Update current pointer to 1 higher, or the first suggestion if the suggestions are exhausted
        self.current_suggestion = (self.current_suggestion + 1) % len(self.suggestions)

        return suggestion


def initialize_state():
    initial_sentence = input("Hello, how may I help you today?")

    food_type, price, area = extract_from_sentence(initial_sentence)
    return State(food_type=food_type, price=price, area=area)


def update_state(state: State, classifier: Classifier, sentence: str):
    # Ask for more info if it is required
    if state.food_type is None or state.price is None or state.area is None:
        new_food_type, new_price, new_area = extract_from_sentence(sentence)
        if new_food_type is not None:
            state.food_type = new_food_type
        if new_price is not None:
            state.price = new_price
        if new_area is not None:
            state.area = new_area
        if state.state_number <= 2 and state.food_type is not None:
            state.state_number = 3
        if state.state_number is 3 and state.price is not None:
            state.state_number = 4
        if state.state_number is 4 and state.area is not None:
            state.state_number = 5

    # Otherwise, check if a request needs to be made
    else:
        bow = load("../models/bow.joblib")
        clf = classifier.load_from_file()
        response_type = "".join(clf.predict(bow.transform([sentence]))) if type(
            clf).__name__ != "RuleBasedClassifier" else "".join(clf.predict([sentence]))

        # If an alternative is requested, no state update is needed
        if state.state_number is 5 and response_type is "reqalts":
            return
        # If a request is made, the state should be updated to give details in the next iteration
        if response_type is "request":
            type = determine_post_or_phone_question(sentence)
            if type is "post":
                state.state_number = 6
            else:
                state.state_number = 7
            return
        state.state_number = 8


def extract_from_sentence(sentence):
    # TODO
    return None, None, None


def determine_post_or_phone_question(sentence):
    # TODO
    return "post"
