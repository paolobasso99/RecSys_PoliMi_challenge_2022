import pandas as pd
from datetime import datetime
import os
import numpy as np

def create_submission(recommender):
    target_users = pd.read_csv('Data/data_target_users_test.csv', dtype={0:int})["user_id"].values

    out = os.path.join('Output', datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.csv')
    print("Creating recomendations...")
    recomendations = recommender.recommend(
        target_users,
        cutoff = 10
    )
    print("Writing to file")
    with open(out, 'w') as f:
        f.write('user_id,item_list')

        for i in range(target_users):
            f.write(str(target_users[i]) + "," + " ".join(str(x) for x in recomendations[i]))
