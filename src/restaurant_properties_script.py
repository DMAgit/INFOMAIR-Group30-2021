import pandas as pd
import numpy as np

res_df = pd.read_csv(r'../data/restaurant_info.csv')  # read the csv so we can work with it


# Food quality column
res_df['food_quality'] = np.random.randint(0, 2, res_df.shape[0])  # set each row to either 0 or 1 (even distribution)
res_df.loc[res_df.food_quality == 1, 'food_quality'] = 'good'  # replace the 1s with 'good'
res_df.loc[res_df.food_quality == 0, 'food_quality'] = None  # replace the 0s with None (emtpy cell)

# Crowdedness
res_df['crowdedness'] = np.random.randint(0, 2, res_df.shape[0])  # set each row to either 0 or 1 (even distribution)
res_df.loc[res_df.crowdedness == 1, 'crowdedness'] = 'busy'  # replace the 1s with 'busy'
res_df.loc[res_df.crowdedness == 0, 'crowdedness'] = None  # replace the 0s with None (emtpy cell)

# Length of stay
res_df['length_stay'] = np.random.randint(0, 2, res_df.shape[0])  # set each row to either 0 or 1 (even distribution)
res_df.loc[res_df.length_stay == 1, 'length_stay'] = 'long'  # replace the 1s with 'long'
res_df.loc[res_df.length_stay == 0, 'length_stay'] = None  # replace the 0s with None (emtpy cell)

res_df.to_csv(r'../data/restaurant_info_properties.csv')
