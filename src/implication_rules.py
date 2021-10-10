def is_busy(suggestion):
    """
    Determines if a suggestion restaurant is busy.
    :param suggestion: Suggestion to classify.
    :return: Boolean indicating business.
    """
    return suggestion.crowdedness == "busy" or (suggestion.food_quality == "good" and suggestion.pricerange == "cheap")


def is_long(suggestion):
    """
    Determines if a suggestion restaurant takes long.
    :param suggestion: Suggestion to classify.
    :return: Boolean indicating length of stay.
    """
    return suggestion.length_stay == "long" or suggestion.food == "spanish" or is_busy(suggestion)


def children_advised(suggestion):
    """
    Determines if it is advised to bring children to a suggestion.
    :param suggestion: Suggestion to classify.
    :return: Boolean indicating child-appropriateness.
    """
    return not is_long(suggestion)


def is_romantic(suggestion):
    """
    Determines if a suggested restaurant is romantic.
    :param suggestion: Suggestion to classify.
    :return: Boolean indicating if the restaurant is romantic.
    """
    # A restaurant being crowded takes priority over length of stay,
    # so if a restaurant is busy, the restaurant will never be romantic
    if is_busy(suggestion):
        return False
    return is_long(suggestion)


def generate_reasoning(state, suggestion):
    """
    Generates a sentence explaining the restaurant choice.
    :param state: Dialog state.
    :param suggestion: Given suggestion.
    :return: String containing the explanation for the suggestion.
    """
    busy_sentence = ""
    child_sentence = ""
    romantic_sentence = ""
    long_sentence = ""
    if state.wants_child is not None:
        if state.wants_child:
            child_sentence = f"{suggestion.restaurantname} is suitable for children.\n"
        else:
            child_sentence = f"{suggestion.restaurantname} is not suitable for children, because the average stay at " \
                             f"this restaurant is quite long.\n"
    if state.wants_busy is not None:
        if state.wants_busy:
            busy_sentence = f"{suggestion.restaurantname} is a busy restaurant.\n"
            if suggestion.crowdedness == "busy":
                busy_sentence = "Our database shows " + busy_sentence
            elif suggestion.food_quality == "good" and suggestion.pricerange == "cheap":
                busy_sentence = "Because this restaurant serves cheap and good food, " + busy_sentence
        else:
            busy_sentence = f"{suggestion.restaurantname} is not a busy restaurant.\n"

    if state.wants_long is not None:
        if state.wants_long:
            long_sentence = f"{suggestion.restaurantname} has long average visit time, this is because"
            if suggestion.food == "spanish":
                long_sentence = long_sentence + " these spanish waiters have a manana manana attitude.\n"
            elif is_busy(suggestion):
                long_sentence = long_sentence + " this restaurant is always full of people.\n"
        else:
            long_sentence = f"{suggestion.restaurantname} is not suitable for long visits.\n"

    if state.wants_romantic is not None:
        if state.wants_romantic:
            romantic_sentence = f"{suggestion.restaurantname} is suitable for romantic occasions, because you can " \
                                f"stay here for a long time.\n"
        else:
            romantic_sentence = f"{suggestion.restaurantname} is not suitable for romantic occasions."
            if is_busy(suggestion) == "busy":
                romantic_sentence = romantic_sentence + " The reason for this is that it is always crowded in" \
                                                        " this restaurant\n"
            else:
                romantic_sentence = "Our database shows " + romantic_sentence + "\n"

    return busy_sentence + child_sentence + romantic_sentence + long_sentence
