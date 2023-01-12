import os
import numpy as np
import scipy.sparse as sps

from skopt.space import Real
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from evaluation.evaluator import EvaluatorHoldout

from data_manager import DatasetLoader, DatasetSplitter, URMGenerator, UIMGenerator
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.MatrixFactorization.IALSRecommenderImplicit import (
    IALSRecommenderImplicit,
)
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.EASE_R.EASE_R_Recommender import (
    EASE_R_Recommender,
)
from Recommenders.BaseRecommender import BaseRecommender

from Recommenders.DataIO import DataIO
from pathlib import Path
from utils.load_best_hyperparameters import load_best_hyperparameters

class Hybrid(BaseRecommender):
    def __init__(self, URM_train, UIM_train, recommenders, verbose=True):
        super().__init__(URM_train, verbose)

        self.UIM_train = UIM_train
        self.recommenders = recommenders

    def _exponential_discounting_scores(self, UIM, alpha, beta):
        if sps.issparse(UIM):
            exp_data: np.ndarray = UIM.data.copy()
        else:
            exp_data = UIM.copy()
        
        exp_data = np.exp(
            self.alpha * exp_data.astype(dtype=np.float32) + self.beta
        )

        mask_pos_inf: np.ndarray = np.isinf(exp_data)
        if mask_pos_inf.any():
            mask_not_pos_inf_on_exp_data_plus_1 = np.logical_not(
                np.isinf(
                    exp_data + 1
                )
            )
            max_value_non_pos_inf = np.nanmax(
                exp_data[mask_not_pos_inf_on_exp_data_plus_1],
            ) + 1

            exp_data[mask_pos_inf] = max_value_non_pos_inf
        exp_data = exp_data.astype(dtype=np.float64)
        
        if sps.issparse(UIM):
            new_sp: sps.csr_matrix = UIM.copy()
            new_sp.data = exp_data

            return new_sp
        
        return exp_data

    def fit(self, alpha, beta, KNN, RP3beta, IALS, EASE_R, SLIMElasticNet):
        self.weights = {
            "KNN": KNN,
            "RP3beta": RP3beta,
            "IALS": IALS,
            "EASE_R": EASE_R,
            "SLIMElasticNet": SLIMElasticNet
        }
        self.alpha = alpha
        self.beta = beta
        self.UIM_scores = self._exponential_discounting_scores(self.UIM_train, self.alpha, self.beta)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if items_to_compute != None:
            print("ALERT: items_to_compute != None")

        item_weights = {}
        for rec_id, rec_obj in self.recommenders.items():
            rec_item_weights = rec_obj._compute_item_score(user_id_array)
            mean = np.mean(rec_item_weights)
            std = np.std(rec_item_weights)
            item_weights[rec_id] = (rec_item_weights - mean) / std

        result = 0
        for rec_id in self.recommenders.keys():
            result += item_weights[rec_id] * self.weights[rec_id]

        result = result.astype(np.float64)
        arr_impressions_discounting_scores: np.ndarray = np.asarray(
        (
            1 + self.UIM_scores[user_id_array, :].toarray()
        )
        ).astype(np.float64)

        new_item_scores = result + result * arr_impressions_discounting_scores

        return new_item_scores


if __name__ == "__main__":
    base_recommenders = {
        "KNN": (ItemKNNCFRecommender, Path("result_experiments/KNN")),
        "RP3beta": (RP3betaRecommender, Path("result_experiments/RP3beta")),
        "IALS": (IALSRecommenderImplicit, Path("result_experiments/IALS")),
        "EASE_R": (EASE_R_Recommender, Path("result_experiments/EASE_R")),
        "SLIMElasticNet": (SLIMElasticNetRecommender, Path("result_experiments/SLIMElasticNet"))
    }
    hyperparameters_range_dictionary = {
        "alpha": Real(-3, -0.001),
        "beta": Real(-2, 2)
    }

    dataset_loader = DatasetLoader()
    dataset_splitter = DatasetSplitter(dataset_loader)
    dataset_train, dataset_val = dataset_splitter.load_interactions_train_val()
    URM_generator = URMGenerator(dataset_train, dataset_val)
    URM_train, URM_val = URM_generator.generate_implicit_URM()
    impressions_train, impressions_val = dataset_splitter.load_impressions_train_val()
    
    UIM_generator = UIMGenerator(impressions_train, impressions_val, URM_train.shape[0], URM_train.shape[1])
    UIM_train, UIM_val = UIM_generator.generate()

    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[10])
    best_weights = load_best_hyperparameters(Path("result_experiments/Hybrid6138"))

    loaded_recommenders = {}
    for recommender_id, (recommender_class, folder) in base_recommenders.items():
        URM_train_file = folder / "tuned_URM/URM_train.npz"
        if URM_train_file.exists():
            recommender_URM_train = sps.load_npz(URM_train_file)
            recommender_obj = recommender_class(recommender_URM_train)
            recommender_obj.load_model(
                str(folder / "tuned_URM"),
                (recommender_class.RECOMMENDER_NAME + "_best_model.zip"),
            )
        else:
            print(f"WARNING: Using implicit URM for {recommender_id}")
            recommender_obj = recommender_class(URM_train)
            recommender_obj.load_model(
                str(folder),
                (recommender_class.RECOMMENDER_NAME + "_best_model.zip"),
            )
            
        loaded_recommenders[recommender_id] = recommender_obj

    output_folder_path = "result_experiments/Hybrid_impressions/"
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
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, UIM_train, loaded_recommenders],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS=best_weights,
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
