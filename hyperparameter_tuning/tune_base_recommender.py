import os
from typing import Dict, Any

from skopt.space import Real, Integer, Categorical

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.MatrixFactorization.IALSRecommenderImplicit import (
    IALSRecommenderImplicit,
)
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from data_manager import DatasetLoader, DatasetSplitter, URMGenerator
from evaluation.evaluator import EvaluatorHoldout
from hyperparameter_tuning.SearchBayesianSkopt import SearchBayesianSkopt
from hyperparameter_tuning.SearchAbstractClass import SearchInputRecommenderArgs
from Recommenders.DataIO import DataIO


def _get_hyperparameters_range(recommender_class: BaseRecommender) -> Dict[str, Any]:
    if recommender_class == EASE_R_Recommender:
        hyperparameters_range_dictionary = {
            "topK": Categorical([None]),
            "normalize_matrix": Categorical([False]),
            "l2_norm": Real(low=1e0, high=1e7, prior="log-uniform"),
        }
    elif recommender_class == IALSRecommenderImplicit:
        hyperparameters_range_dictionary = {
            "num_factors": Integer(1, 500),
            "epochs": Integer(100, 200),
            "alpha": Real(low=1e-3, high=50.0, prior="log-uniform"),
            "reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
        }
    elif recommender_class == ItemKNNCFRecommender:
        hyperparameters_range_dictionary = {
            "topK": Integer(5, 2000),
            "shrink": Integer(0, 500),
            "similarity": Categorical(["cosine"]),
            "normalize": Categorical([True]),
        }
    elif recommender_class == RP3betaRecommender:
        hyperparameters_range_dictionary = {
            "topK": Integer(5, 1000),
            "alpha": Real(low=0, high=2, prior="uniform"),
            "beta": Real(low=0, high=2, prior="uniform"),
            "normalize_similarity": Categorical([True, False]),
        }
    elif recommender_class == SLIMElasticNetRecommender:
        hyperparameters_range_dictionary = {
            "topK": Integer(500, 2000),
            "l1_ratio": Real(low=1e-4, high=0.1, prior="log-uniform"),
            "alpha": Real(low=1e-4, high=0.1, prior="uniform"),
        }
    else:
        return ValueError(f"The recommender class {recommender_class} is not supported")
    return hyperparameters_range_dictionary


def tune_base_recommender(
    recommender_class: BaseRecommender,
    output_folder: str,
    n: int = 100,
    save_trained_on_all: bool = True,
):
    if not os.path.exists("results_experiments"):
        os.makedirs("result_experiments")

    full_output_path = "results_experiments/" + output_folder
    if not os.path.exists(full_output_path):
        os.makedirs(full_output_path)

    # Load data
    dataset_loader = DatasetLoader()
    dataset_splitter = DatasetSplitter(dataset_loader)
    dataset_train, dataset_val = dataset_splitter.load_interactions_train_val()
    URM_generator = URMGenerator(dataset_train, dataset_val)
    URM_train, URM_val = URM_generator.generate_explicit_URM(
        log_base=4, views_weight=1, details_weight=0.8
    )
    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[10])

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
    hyperparameters_range_dictionary = _get_hyperparameters_range(recommender_class)

    hyperparameter_search.search(
        recommender_input_args,
        hyperparameter_search_space=hyperparameters_range_dictionary,
        n_cases=n,
        n_random_starts=int(n * 0.3),
        save_model="best",
        output_folder_path=full_output_path,  # Where to save the results
        output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
        metric_to_optimize="MAP",
        cutoff_to_optimize=10,
    )

    data_loader = DataIO(folder_path=full_output_path)
    search_metadata = data_loader.load_data(
        recommender_class.RECOMMENDER_NAME + "_metadata.zip"
    )
    best_hyperparameters = search_metadata["hyperparameters_best"]
    print("best_param", best_hyperparameters)

    if save_trained_on_all:
        recommender: BaseRecommender = recommender_class(URM_train + URM_val)
        recommender.fit(**best_hyperparameters)
        recommender.save_model(
            folder_path=full_output_path,
            file_name=recommender_class.RECOMMENDER_NAME
            + "_best_model_trained_on_everything.zip",
        )
