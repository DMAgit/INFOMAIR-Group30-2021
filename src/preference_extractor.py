import re
import sys
import pandas as pd
import Levenshtein as lev


def get_test_utterances():
    test_utterances = ["I'm looking for world food", "I want a restaurant that serves world food",
                       "I want a restaurant serving Swedish food", "I'm looking for a restaurant in the center",
                       "I would like a cheap restaurant in the west part of town",
                       "I'm looking for a moderately priced restaurant in the west part of town",
                       "I'm looking for a restaurant in any area that serves Tuscan food",
                       "Can I have an expensive restaurant",
                       "I'm looking for an expensive restaurant and it should serve international food",
                       "I need a Cuban restaurant that is moderately priced",
                       "I'm looking for a moderately priced restaurant with Catalan food",
                       "What is a cheap restaurant in the south part of town", "What about Chinese food",
                       "I wanna find a cheap restaurant", "I'm looking for Persian food please",
                       "Find a Cuban restaurant in the center"]
    return test_utterances


def get_possible_preferences():
    df = pd.read_csv("../data/restaurant_info.csv")
    foods = set(df["food"].tolist())
    areas = set(df["area"].tolist())
    areas = {x for x in areas if pd.notna(x)}
    price_ranges = set(df["pricerange"].tolist())

    return foods, areas, price_ranges


def get_closest_levenshtein(word, possible_words):
    result = None
    min_distance = sys.maxsize
    for possible_word in possible_words:
        distance = lev.distance(word, possible_word)
        if distance < min_distance:
            result = possible_word
            min_distance = distance
    return result


def extract_preferences_from_sentence(sentence):
    food_pattern = re.compile(r"(\w+)\s*food")
    area_pattern = re.compile(r"\b(center|north|east|south|west)\b")
    price_pattern = re.compile(r"\b(cheap\w*|moderat\w*|expensiv\w*)\b")

    sentence = sentence.lower()
    food_matched = food_pattern.search(sentence)
    area_matched = area_pattern.search(sentence)
    price_matched = price_pattern.search(sentence)

    possible_foods, possible_areas, possible_price_ranges = get_possible_preferences()
    food = None if food_matched is None else get_closest_levenshtein(food_matched.group(1), possible_foods)
    area = None if area_matched is None else get_closest_levenshtein(area_matched.group(1), possible_areas)
    price_range = None if price_matched is None else get_closest_levenshtein(price_matched.group(1),
                                                                             possible_price_ranges)

    return food, area, price_range


for utterance in get_test_utterances():
    f, a, p = extract_preferences_from_sentence(utterance)
    print(utterance)
    print(f"\tFood: {f}, area: {a}, price: {p}")
