import os
import numpy as np
import scipy.sparse as sps

from skopt.space import Real
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from evaluation.evaluator import EvaluatorHoldout

from data_manager import DatasetLoader, DatasetSplitter, URMGenerator
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.MatrixFactorization.IALSRecommenderImplicit import (
    IALSRecommenderImplicit,
)
from Recommenders.EASE_R.EASE_R_Recommender import (
    EASE_R_Recommender,
)
from Recommenders.BaseRecommender import BaseRecommender

from Recommenders.DataIO import DataIO
from typing import Dict, Any, Tuple
from pathlib import Path


class Hybrid(BaseRecommender):
    def __init__(self, URM_train, recommenders, verbose=True):
        super().__init__(URM_train, verbose)

        self.recommenders = recommenders

    def fit(self, KNN, RP3beta, IALS, EASE_R):
        self.weights = {
            "KNN": KNN,
            "RP3beta": RP3beta,
            "IALS": IALS,
            "EASE_R": EASE_R,
        }

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights = {}
        for rec_id, rec_obj in self.recommenders.items():
            print(f"Computing item weights for {rec_id}")
            rec_item_weights = rec_obj._compute_item_score(user_id_array)
            mean = np.mean(rec_item_weights)
            std = np.std(rec_item_weights)
            item_weights[rec_id] = (rec_item_weights - mean) / std

        print("Computing aggregate item weights")
        result = 0
        for rec_id in self.recommenders.keys():
            result += item_weights[rec_id] * self.weights[rec_id]

        return result


if __name__ == "__main__":
    base_recommenders = {
        "KNN": (ItemKNNCFRecommender, Path("result_experiments/KNN")),
        "RP3beta": (RP3betaRecommender, Path("result_experiments/RP3beta")),
        # "IALS": (IALSRecommenderImplicit, Path("result_experiments/IALS")),
        "EASE_R": (EASE_R_Recommender, Path("result_experiments/EASE_R")),
    }

    hyperparameters_range_dictionary = {
        "KNN": Real(0.0, 1.0),
        "RP3beta": Real(0.2, 1.0),
        "IALS": Real(0.0, 1.0),
        "EASE_R": Real(0.3, 1.0),
    }

    dataset_loader = DatasetLoader()
    dataset_splitter = DatasetSplitter(dataset_loader)
    dataset_train, dataset_val = dataset_splitter.load_train_val()
    URM_generator = URMGenerator(dataset_train, dataset_val)
    URM_train, URM_val = URM_generator.generate_implicit_URM()
    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[10])

    loaded_recommenders = {}
    for recommender_id, (recommender_class, folder) in base_recommenders.items():
        recommender_URM_train = sps.load_np(folder / "tuned_URM/URM_train.npz")
        recommender_obj = recommender_class(recommender_URM_train)
        recommender_obj.load_model(
            str(folder / "tuned_URM"),
            (recommender_class.RECOMMENDER_NAME + "_best_model.zip"),
        )
        loaded_recommenders[recommender_id] = recommender_obj

    output_folder_path = "result_experiments/Hybrid/"
    recommender_class = Hybrid
    n_cases = 100
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    hyperparameter_search = SearchBayesianSkopt(
        recommender_class,
        evaluator_validation=evaluator,
    )

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, loaded_recommenders],
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
        save_model="no",
        output_folder_path=output_folder_path,  # Where to save the results
        output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
        metric_to_optimize=metric_to_optimize,
        cutoff_to_optimize=cutoff_to_optimize,
    )

    data_loader = DataIO(folder_path=output_folder_path)
    search_metadata = data_loader.load_data(
        recommender_class.RECOMMENDER_NAME + "_metadata.zip"
    )

    result_on_validation_df = search_metadata["result_on_validation_df"]
    print(result_on_validation_df)
    best_hyperparameters = search_metadata["hyperparameters_best"]
    print(best_hyperparameters)
