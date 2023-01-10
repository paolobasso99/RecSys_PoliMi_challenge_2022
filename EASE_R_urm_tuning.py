import os
import random

import numpy as np
from Recommenders.EASE_R.EASE_R_Recommender import (
    EASE_R_Recommender,
)
from utils.URM_tuner import URM_tuner

if __name__ == "__main__":
    # Random seed for reproducibility
    SEED = 42
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)

    recommender_class = EASE_R_Recommender
    model_folder = "result_experiments/EASE_R_Recommender4/"

    URM_tuner(100, recommender_class, model_folder)