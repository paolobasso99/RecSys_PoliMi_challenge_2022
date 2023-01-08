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
    URM_train, URM_val = URM_generator.generate_explicit_URM(log_base=17, views_weight=1, details_weight=1)
    URM_all = URM_train + URM_val

    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[10])
    
    output_folder_path = "result_experiments/EASE_R_Recommender2/"
    recommender_class = EASE_R_Recommender
    n_cases = 10
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Define hyperparameters
    hyperparameters_range_dictionary = {
        "topK": Categorical([None]),
        "normalize_matrix": Categorical([False]),
        "l2_norm": Real(low=1e0, high=1e7, prior="log-uniform"),
    }

    hyperparameter_search = SearchBayesianSkopt(
        recommender_class,
        evaluator_validation=evaluator,
    )

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[
            URM_train,
        ],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    hyperparameter_search.search(
        recommender_input_args,
        hyperparameter_search_space=hyperparameters_range_dictionary,
        n_cases=n_cases,
        n_random_starts=n_random_starts,
        save_model="best",
        output_folder_path=output_folder_path,  # Where to save the results
        output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
        metric_to_optimize=metric_to_optimize,
        cutoff_to_optimize=cutoff_to_optimize,
    )