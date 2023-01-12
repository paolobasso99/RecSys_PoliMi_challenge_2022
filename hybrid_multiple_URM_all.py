import numpy as np
import scipy.sparse as sps
from data_manager import DatasetLoader, DatasetSplitter, URMGenerator
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.MatrixFactorization.IALSRecommenderImplicit import (
    IALSRecommenderImplicit,
)
from Recommenders.EASE_R.EASE_R_Recommender import (
    EASE_R_Recommender,
)
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from utils.load_best_hyperparameters import load_best_hyperparameters
from utils.create_submission import create_submission
from pathlib import Path


class Hybrid(BaseRecommender):
    def __init__(self, URM_train, recommenders, verbose=True):
        super().__init__(URM_train, verbose)

        self.recommenders = recommenders

    def fit(self, KNN, RP3beta, IALS, EASE_R, SLIMElasticNet):
        self.weights = {
            "KNN": KNN,
            "RP3beta": RP3beta,
            "IALS": IALS,
            "EASE_R": EASE_R,
            "SLIMElasticNet": SLIMElasticNet
        }

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights = {}
        for rec_id, rec_obj in self.recommenders.items():
            rec_item_weights = rec_obj._compute_item_score(user_id_array)
            mean = np.mean(rec_item_weights)
            std = np.std(rec_item_weights)
            item_weights[rec_id] = (rec_item_weights - mean) / std

        result = 0
        for rec_id in self.recommenders.keys():
            result += item_weights[rec_id] * self.weights[rec_id]

        return result

if __name__ == "__main__":
    base_recommenders = {
        "KNN": (ItemKNNCFRecommender, Path("result_experiments/KNN")),
        "RP3beta": (RP3betaRecommender, Path("result_experiments/RP3beta")),
        "IALS": (IALSRecommenderImplicit, Path("result_experiments/IALS")),
        "EASE_R": (EASE_R_Recommender, Path("result_experiments/EASE_R")),
        "SLIMElasticNet": (SLIMElasticNetRecommender, Path("result_experiments/SLIMElasticNet"))
    }

    dataset_loader = DatasetLoader()
    dataset_splitter = DatasetSplitter(dataset_loader)
    dataset_train, dataset_val = dataset_splitter.load_interactions_train_val()
    URM_generator = URMGenerator(dataset_train, dataset_val)
    URM_train, URM_val = URM_generator.generate_implicit_URM()

    loaded_recommenders = {}
    for recommender_id, (recommender_class, folder) in base_recommenders.items():
        URM_train_file = folder / "tuned_URM/URM_train.npz"
        if URM_train_file.exists():
            recommender_URM_train = sps.load_npz(URM_train_file)
            recommender_URM_val = sps.load_npz(folder / "tuned_URM/URM_val.npz")
            recommender_obj = recommender_class(recommender_URM_train + recommender_URM_val)
            recommender_obj.load_model(
                str(folder / "tuned_URM"),
                (recommender_class.RECOMMENDER_NAME + "_best_model_trained_on_everything.zip"),
            )
        else:
            print(f"WARNING: Using implicit URM for {recommender_id}")
            recommender_obj = recommender_class(URM_train + URM_val)
            recommender_obj.load_model(
                str(folder),
                (recommender_class.RECOMMENDER_NAME + "_best_model_trained_on_everything.zip"),
            )
            
        loaded_recommenders[recommender_id] = recommender_obj

    best_hyperparameters = load_best_hyperparameters(Path("result_experiments/Hybrid"))
    recommender = Hybrid(URM_train + URM_val, loaded_recommenders)
    recommender.fit(**best_hyperparameters)
    create_submission(recommender)