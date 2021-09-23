# DEPRECATED
# this has been implemented in state_manager in the generate_suggestions function
# but gonna keep it for now

import pandas as pd
from Levenshtein import distance

# reading the csv
res_df = pd.read_csv(r'../data/restaurant_info.csv')


# print(res_df)


def get_restaurant(res_df: pd.DataFrame, food_type, price, area):
    matches_df = res_df.copy(deep=True)  # at the start all restaurants are matches

    if food_type is not None:  # check if the user has specified a food type
        if food_type in matches_df['food']:  # check if there is an exact match
            matches_df = matches_df[matches_df['food'].str.contains(food_type)]  # filter to only that food type
        else:  # if not we want to iterate all of the food types and compute the Levenshtein distance
            for i in matches_df.loc[:, 'food']:
                if distance(food_type, i) >= 2:  # >= 2 is an arbitrary cut-off point, we could do sth fancy instead
                    matches_df = matches_df[matches_df['food'] != i]

    if price is not None:  # check if the user has specified a price range
        if price in matches_df['pricerange']:  # check if there is an exact match
            matches_df = matches_df[matches_df['pricerange'].str.contains(price)]
        else:
            for i in matches_df.loc[:, 'pricerange']:
                if distance(price, i) >= 2:  # >= 2 is an arbitrary cut-off point, we could do sth fancy instead
                    matches_df = matches_df[matches_df['pricerange'] != i]

    if area is not None:  # check if the user has specified an area
        if area in matches_df['area']:
            matches_df = matches_df[matches_df['area'].str.contains(area)]  # filter to only that area
        else:
            for i in matches_df.loc[:, 'area']:
                if distance(area, i) >= 2:  # >= 2 is an arbitrary cut-off point, we could do sth fancy instead
                    matches_df = matches_df[matches_df['area'] != i]

    # please someone make sure we don't need these
    # remove the features we don't need
    matches_df.drop(labels=['pricerange', 'area', 'food', 'addr'], axis=1, inplace=True)
    matches_df = matches_df.to_numpy()  # convert from dataframe to an array
    return matches_df


with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(get_restaurant(res_df, 'italian', 'cheap', None))
