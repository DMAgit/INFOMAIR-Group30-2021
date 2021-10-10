import re
import pandas as pd
from Levenshtein import distance


def get_possible_preferences():
    """
    Loads all the possible vales from each set of preferences from the available data

    :return: foods: set - all the valid cuisines
                areas: set - all the valid areas
                price_ranges: set - all the valid price ranges
    """
    df = pd.read_csv("../data/restaurant_info.csv")
    foods = set(list(df["food"]))
    areas = set(list(df["area"].dropna()))
    price_ranges = set(list(df["pricerange"]))

    return foods, areas, price_ranges


def is_food_in_set(sentence, possible_foods):
    """
    Given a sentence checks if it contains a valid food type

    :param sentence: str - the sentence to check
    :param possible_foods: set - all the valid cuisines
    :return: token: str - the food type found / None if no cuisine on the sentence
    """
    for token in sentence.split(" "):
        if token in possible_foods:
            return token
    return None


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


def process_preference_type(preference, possible_preferences, threshold):
    """
    Given a preference as a string checks using levenshtein distance if it is on the set of possible values for
    that preference

    :param preference: str - the preference (food, area, price)
    :param possible_preferences: set - the possible valid values for that preference
    :param threshold: int - the maximum distance allowed when applying levenshtein
    :return: preference: str - if it is directly contained on the set
                candidate: str - the most similar preference found
                None - if there is no match at all
    """
    candidate, min_distance = get_closest_levenshtein(preference, possible_preferences, threshold)
    if preference in possible_preferences:
        return preference
    elif candidate in possible_preferences:
        return candidate
    return None


def extract_preferences_from_sentence(sentence, lev_threshold, state_number):
    """
    Function that given a sentence applies pattern matching to extract user preferences taking into account the
    current state of the dialog manager. Uses levenshtein distance to find the most similar preferences is there
    is not an exact match. Also processes the cases when the user does not care about a preference.

    :param sentence: str - the sentence to extract the preferences from
    :param lev_threshold: int - the maximum distance allowed when applying levenshtein
    :param state_number: int - the current state of the dialog manager (used to check for the "do not care" utterances)
    :return: food, area, price_range: str - the preferences extracted (None if no match)
    """
    # Matches <type> food and <type> restaurant
    food_pattern = re.compile(r"(\w+)\s*(?:food|restaurant)")
    # Matches any mention of cardinality
    area_pattern = re.compile(r"\b(center|north|east|south|west)\b")
    # Matches any mention of the price ranges (also adverbs, just in case)
    price_pattern = re.compile(r"\b(cheap\w*|moderate\w*|expensive\w*)\b")
    # Don't care pattern
    dontcare_pattern = re.compile(r"(?:i|it)?\s*(any\s*(?:kind)?|dont|don't|do not\s*(?:mind|care)|doesnt\s*matter)\b")

    sentence = sentence.lower()
    food_matched = food_pattern.findall(sentence)
    area_matched = area_pattern.findall(sentence)
    price_matched = price_pattern.findall(sentence)
    dontcare_matched = dontcare_pattern.findall(sentence)

    possible_foods, possible_areas, possible_price_ranges = get_possible_preferences()

    # Food processing
    if not food_matched:  # No match at all (try to look for the name directly)
        food = is_food_in_set(sentence, possible_foods)
        # Return the 'dontcare' token if the user doesn't have preference about the food
        if food is None and state_number <= 2:
            food = "dontcare" if dontcare_matched else None
    else:
        # Preprocess: change 'world' for 'international'
        food_matched = list(map(lambda f: "international" if f == "world" else f, food_matched))
        # Look for the most possible restaurant
        food_matched = list(map(lambda f: process_preference_type(f, possible_foods, lev_threshold), food_matched))
        food = food_matched[0] if food_matched else None

    # Area processing
    if not area_matched:  # No match at all
        area = None
        # Return the 'dontcare' token if the user doesn't have preference about the area
        if state_number == 4:
            area = "dontcare" if dontcare_matched else None
    else:
        # Look for the most possible area
        area_matched = list(map(lambda a: process_preference_type(a, possible_areas, lev_threshold), area_matched))
        area = area_matched[0] if area_matched else None

    # Price processing
    if not price_matched:  # No match at all
        price_range = None
        # Return the 'dontcare' token if the user doesn't have preference about the price_range
        if state_number == 3:
            price_range = "dontcare" if dontcare_matched else None
    else:
        # Look for the most possible area
        price_matched = list(map(lambda p: process_preference_type(p, possible_price_ranges, lev_threshold),
                                 price_matched))
        price_range = price_matched[0] if price_matched else None

    return food, area, price_range


def extract_post_or_phone(sentence):
    """
    Function that determines if the user is requesting the phone number or the postal code of a restaurant applying
    pattern matching. In this case a default and tested value for the levenshtein threshold is used (4).

    :param sentence: str - the sentence to analyse
    :return: result: str - phone/post or None if no match
    """
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


def extract_additional(sentence):
    """
    Function that applies pattern matching to determine the additional characteristics of a restaurant: busyness, long
    stay, romantic or children friendly.

    :param sentence: str - the sentence to extract the preferences from
    :return: busy, length, children, romantic: bool - if the sentences matches any of the preferences described above
    """
    busy = None
    length = None
    children = None
    romantic = None

    unromantic_pattern = re.compile(r"(no(t|n) (un)*romantic)")
    romantic_pattern = re.compile(r"(romantic)")
    if unromantic_pattern.findall(sentence):
        romantic = False
    elif romantic_pattern.findall(sentence):
        romantic = True

    quiet_pattern = re.compile(r"(no(t|n) busy)|(quiet)")
    busy_pattern = re.compile(r"(busy)|(crowded)")
    if quiet_pattern.findall(sentence):
        busy = False
    elif busy_pattern.findall(sentence):
        busy = True

    paedophobia_pattern = re.compile(r"(no(t|n)? (child(ren)?)|(kids?)|(baby|(ies)))")
    kids_pattern = re.compile(r"(child(ren)?)|(kids?)|(baby|(ies))")
    if paedophobia_pattern.findall(sentence):
        children = False
    elif kids_pattern.findall(sentence):
        children = True

    short_pattern = re.compile(r"(short)")
    long_pattern = re.compile(r"(long)")
    if short_pattern.findall(sentence):
        length = False
    elif long_pattern.findall(sentence):
        length = True

    return busy, length, children, romantic
