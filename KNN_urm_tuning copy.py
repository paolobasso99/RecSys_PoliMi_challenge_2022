import os
import random

import numpy as np
from Recommenders.KNN.ItemKNNCFRecommender import (
    ItemKNNCFRecommender,
)
from utils.URM_tuner import URM_tuner

if __name__ == "__main__":
    # Random seed for reproducibility
    SEED = 42
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)

    recommender_class = ItemKNNCFRecommender
    model_folder = "result_experiments/KNN/"

    URM_tuner(100, recommender_class, model_folder)