import os
import numpy as np

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

from Recommenders.DataIO import DataIO


class Hybrid(BaseRecommender):
    def __init__(self, URM_train, verbose=True):
        super().__init__(URM_train, verbose)

        self.knn = ItemKNNCFRecommender(URM_train)
        self.rp3 = RP3betaRecommender(URM_train)
        self.ials = IALSRecommenderImplicit(URM_train)
        self.top_pop = TopPop(URM_train)
        self.ease_r = EASE_R_Recommender(URM_train)

    def fit(self, knn_w, rp3_w, ials_w, top_pop_w, ease_r_w):
        self.weights = {
            "knn": knn_w,
            "rp3": rp3_w,
            "ials": ials_w,
            "top_pop": top_pop_w,
            "ease_r": ease_r_w,
        }
        print(self.weights)
        self.knn.load_model(
            "result_experiments/ItemKNNCFRecommender/",
            "ItemKNNCFRecommender_best_model.zip",
        )
        self.rp3.load_model(
            "result_experiments/RP3betaRecommender/",
            "RP3betaRecommender_best_model.zip",
        )
        self.ials.load_model(
            "result_experiments/IALSRecommenderImplicit/",
            "IALSRecommenderImplicit_best_model.zip",
        )
        self.ease_r.load_model(
            "result_experiments/EASE_R_Recommender/",
            "EASE_R_Recommender_best_model.zip",
        )
        self.top_pop.fit()

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.knn._compute_item_score(user_id_array)
        item_weights_2 = self.rp3._compute_item_score(user_id_array)
        item_weights_3 = self.ials._compute_item_score(user_id_array)
        item_weights_4 = self.top_pop._compute_item_score(user_id_array)
        item_weights_5 = self.ease_r._compute_item_score(user_id_array)

        mean1 = np.mean(item_weights_1)
        mean2 = np.mean(item_weights_2)
        mean3 = np.mean(item_weights_3)
        mean4 = np.mean(item_weights_4)
        mean5 = np.mean(item_weights_5)
        std1 = np.std(item_weights_1)
        std2 = np.std(item_weights_2)
        std3 = np.std(item_weights_3)
        std4 = np.std(item_weights_4)
        std5 = np.std(item_weights_5)
        item_weights_1 = (item_weights_1 - mean1) / std1
        item_weights_2 = (item_weights_2 - mean2) / std2
        item_weights_3 = (item_weights_3 - mean3) / std3
        item_weights_4 = (item_weights_4 - mean4) / std4
        item_weights_5 = (item_weights_5 - mean5) / std5

        item_weights = (
            item_weights_1 * self.weights["knn"]
            + item_weights_2 * self.weights["rp3"]
            + item_weights_3 * self.weights["ials"]
            + item_weights_4 * self.weights["top_pop"]
            + item_weights_5 * self.weights["ease_r"]
        )

        for i, user_id in enumerate(user_id_array):
            relevant_items = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
            if len(relevant_items) < 1:
                item_weights[i] = item_weights_4

        return item_weights


if __name__ == "__main__":
    dataset_loader = DatasetLoader()
    dataset_splitter = DatasetSplitter(dataset_loader)
    dataset_train, dataset_val = dataset_splitter.load_train_val()
    URM_generator = URMGenerator(dataset_train, dataset_val)
    URM_train, URM_val = URM_generator.generate_explicit_URM()
    URM_all = URM_train + URM_val

    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[10])

    output_folder_path = "result_experiments/Hybrid_knn_rp3_ials_toppop_easer/"
    recommender_class = Hybrid
    n_cases = 30
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Define hyperparameters
    hyperparameters_range_dictionary = {
        "knn_w": Real(0.0, 1.0),
        "rp3_w": Real(0.0, 1.0),
        "ials_w": Real(0.0, 1.0),
        "top_pop_w": Real(0.0, 0.5),
        "ease_r_w": Real(0.0, 1.0),
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

    """ hyperparameter_search.search(
        recommender_input_args,
        hyperparameter_search_space=hyperparameters_range_dictionary,
        n_cases=n_cases,
        n_random_starts=n_random_starts,
        save_model="best",
        output_folder_path=output_folder_path,  # Where to save the results
        output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
        metric_to_optimize=metric_to_optimize,
        cutoff_to_optimize=cutoff_to_optimize,
    ) """

    data_loader = DataIO(folder_path=output_folder_path)
    search_metadata = data_loader.load_data(
        recommender_class.RECOMMENDER_NAME + "_metadata.zip"
    )

    result_on_validation_df = search_metadata["result_on_validation_df"]
    print(result_on_validation_df)
    best_hyperparameters = search_metadata["hyperparameters_best"]
    print(best_hyperparameters)
