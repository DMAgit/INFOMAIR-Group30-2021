import re
import pandas as pd
from Levenshtein import distance


def get_test_utterances():
    return ["I'm looking for world food", "I want a restaurant that serves world food",
            "I want a restaurant serving Swedish food", "I'm looking for a restaurant in the center",
            "I would like a cheap restaurant in the west part of town",
            "I'm looking for a moderately priced restaurant in the west part of town",
            "I'm looking for a restaurant in any area that serves Tuscan food", "Can I have an expensive restaurant",
            "I'm looking for an expensive restaurant and it should serve international food",
            "I need a Cuban restaurant that is moderately priced",
            "I'm looking for a moderately priced restaurant with Catalan food",
            "What is a cheap restaurant in the south part of town", "What about Chinese food",
            "I wanna find a cheap restaurant", "I'm looking for Persian food please",
            "Find a Cuban restaurant in the center"]


def get_possible_preferences():
    df = pd.read_csv("../data/restaurant_info.csv")
    foods = set(list(df["food"]))
    areas = set(list(df["area"].dropna()))
    price_ranges = set(list(df["pricerange"]))

    return foods, areas, price_ranges


def get_closest_levenshtein(word, possible_words, threshold):
    """
    Returns the closest word from a corpus of possible words given the other word to check. It also applies a
    similarity threshold to not introduce too unrelated words (not derived from misspellings)

    :param word: str - the word to check
    :param possible_words: list<str> - corpus of candidate words
    :param threshold: int - max levenshtein distance allowed
    :return: result: str - the most similar word found (None, if above the threshold)
                min_distance: int - the distance that was needed to find the similar word
    """
    result = None
    min_distance = 10
    for possible_word in possible_words:
        word_distance = distance(word, possible_word)
        if word_distance < min_distance:
            result = possible_word
            min_distance = word_distance
    result = result if min_distance < threshold else None
    return result, min_distance


def is_food_in_set(sentence, possible_foods):
    for token in sentence.split(" "):
        if token in possible_foods:
            return token
    return None


def process_preference_type(preference, possible_preferences, threshold):
    candidate, min_distance = get_closest_levenshtein(preference, possible_preferences, threshold)
    if preference in possible_preferences:
        return preference
    elif candidate in possible_preferences:
        return candidate
    return None


def extract_preferences_from_sentence(sentence):
    # Matches <type> food and <type> restaurant
    food_pattern = re.compile(r"(any kind of\s*|any\s*|\w+)\s*(?:food|restaurant)")
    # Matches any mention of cardinality
    area_pattern = re.compile(r"\b(any area|center|north|east|south|west)\b")
    # Matches any mention of the price ranges (also adverbs, just in case)
    price_pattern = re.compile(r"\b(any price|cheap\w*|moderat\w*|expensiv\w*)\b")

    sentence = sentence.lower()
    food_matched = food_pattern.findall(sentence)
    area_matched = area_pattern.findall(sentence)
    price_matched = price_pattern.findall(sentence)

    possible_foods, possible_areas, possible_price_ranges = get_possible_preferences()

    # Food processing
    if not food_matched:  # No match at all
        food = is_food_in_set(sentence, possible_foods)  # Look directly for cuisine name
    else:
        # Return the 'dontcare' token if the user doesn't have preference about the cuisine
        if any("any" in s for s in food_matched):
            food = "dontcare"
        else:
            # Preprocess: change 'world' for 'international'
            food_matched = list(map(lambda f: "international" if f == "world" else f, food_matched))
            # Look for the most possible restaurant
            food_matched = list(map(lambda f: process_preference_type(f, possible_foods, 3), food_matched))
            food = food_matched[0] if food_matched else None

    # Area processing
    if not area_matched:  # No match at all
        area = None
    else:
        # Return the 'dontcare' token if the user doesn't have preference about the area
        if "any" in area_matched[0]:
            area = "dontcare"
        else:
            # Look for the most possible area
            area_matched = list(map(lambda a: process_preference_type(a, possible_areas, 3), area_matched))
            area = area_matched[0] if area_matched else None

    # Price processing
    if not price_matched:  # No match at all
        price_range = None
    else:
        # Return the 'dontcare' token if the user doesn't have preference about the price_range
        if "any" in price_matched[0]:
            price_range = "dontcare"
        else:
            # Look for the most possible area
            price_matched = list(map(lambda p: process_preference_type(p, possible_price_ranges, 3), price_matched))
            price_range = price_matched[0] if price_matched else None

    return food, area, price_range


def extract_post_or_phone(sentence):
    post_pattern = re.compile(r"(postal\s*|post\s*)(?:code)?")
    phone_pattern = re.compile(r"(telephone\s*|phone\s*)(?:number)?")
    post_matched = post_pattern.findall(sentence)
    phone_matched = phone_pattern.findall(sentence)
    if not post_matched and not phone_matched:  # No match at all
        result = None
    else:
        post_matched = list(map(lambda p: process_preference_type(p, {"post", "phone"}, 4), post_matched))
        phone_matched = list(map(lambda p: process_preference_type(p, {"post", "phone"}, 4), phone_matched))
        result = post_matched[0] if len(post_matched) != 0 \
            else phone_matched[0] if len(phone_matched) != 0 else None

    return result
