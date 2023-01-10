import os
import random

import numpy as np
import pandas as pd
import scipy.sparse as sps

from data_manager import DatasetLoader, DatasetSplitter, URMGenerator
from evaluation.evaluator import EvaluatorHoldout
from Recommenders.EASE_R.EASE_R_Recommender import (
    EASE_R_Recommender,
)
from skopt.space import Real, Integer, Categorical
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from Recommenders.DataIO import DataIO

if __name__ == "__main__":
    # Random seed for reproducibility
    SEED = 42
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    
    dataset_loader = DatasetLoader()
    dataset_splitter = DatasetSplitter(dataset_loader)
    dataset_train, dataset_val = dataset_splitter.load_train_val()
    URM_generator = URMGenerator(dataset_train, dataset_val)
    URM_train, URM_val = URM_generator.generate_explicit_URM(log_base=4, views_weight=1, details_weight=0.8)
    URM_all = URM_train + URM_val

    rec = EASE_R_Recommender(URM_all)
    output_folder_path = "result_experiments/EASE_R/"
    rec.load_model(
        "result_experiments/EASE_R/",
        rec.RECOMMENDER_NAME + "_best_model_trained_on_everything.zip"
    )

    from utils.create_submission import create_submission

    create_submission(rec)