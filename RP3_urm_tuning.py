import os
import random

import numpy as np
from Recommenders.GraphBased.RP3betaRecommender import (
    RP3betaRecommender,
)
from utils.URM_tuner import URM_tuner

if __name__ == "__main__":
    # Random seed for reproducibility
    SEED = 42
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)

    recommender_class = RP3betaRecommender
    model_folder = "result_experiments/RP3/"

    URM_tuner(100, recommender_class, model_folder)