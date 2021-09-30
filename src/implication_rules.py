
def is_busy(suggestion):
    if suggestion.crowdedness == "busy":
        return True
    if suggestion.food_quality == "good" and suggestion.pricerange == "cheap":
        return True
    return False


def is_long(suggestion):
    if suggestion.length_stay == "long":
        return True
    if suggestion.food == "spanish":
        return True
    if is_busy(suggestion):
        return True
    return False


def children_advised(suggestion):
    if is_long(suggestion):
        return False
    return True


def is_romantic(suggestion):
    # crowdedness takes priority over length of stay, so if a restaurant is busy, the restaurant will never be romantic
    if is_busy(suggestion):
        return False
    if is_long(suggestion):
        return True

    return False

