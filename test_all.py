import numpy as np

from data_manager import DatasetLoader, DatasetSplitter, URMGenerator
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.MatrixFactorization.IALSRecommenderImplicit import (
    IALSRecommenderImplicit,
)

from utils.create_submission import create_submission


class Hybrid(BaseRecommender):
    RECOMMENDER_NAME = "Hybrid_KNN_RP3_IALS_TOPPOP"
    def __init__(self, URM_train, verbose=True):
        super().__init__(URM_train, verbose)

        self.knn = ItemKNNCFRecommender(URM_train)
        self.rp3 = RP3betaRecommender(URM_train)
        self.ials = IALSRecommenderImplicit(URM_train)
        self.top_pop = TopPop(URM_train)

    def fit(self, knn_w, rp3_w, ials_w, top_pop_w):
        self.weights = {
            "knn": knn_w,
            "rp3": rp3_w,
            "ials": ials_w,
            "top_pop": top_pop_w,
        }
        print(self.weights)
        self.knn.load_model(
            "result_experiments/ItemKNNCFRecommender/",
            "ItemKNNCFRecommender_best_model_trained_on_everything.zip",
        )
        self.rp3.load_model(
            "result_experiments/RP3betaRecommender/",
            "RP3betaRecommender_best_model_trained_on_everything.zip",
        )
        self.ials.load_model(
            "result_experiments/IALSRecommenderImplicit/",
            "IALSRecommenderImplicit_best_model_trained_on_everything.zip",
        )
        self.top_pop.fit()

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.knn._compute_item_score(user_id_array)
        item_weights_2 = self.rp3._compute_item_score(user_id_array)
        item_weights_3 = self.ials._compute_item_score(user_id_array)
        item_weights_4 = self.top_pop._compute_item_score(user_id_array)

        mean1 = np.mean(item_weights_1)
        mean2 = np.mean(item_weights_2)
        mean3 = np.mean(item_weights_3)
        mean4 = np.mean(item_weights_4)
        std1 = np.std(item_weights_1)
        std2 = np.std(item_weights_2)
        std3 = np.std(item_weights_3)
        std4 = np.std(item_weights_4)
        item_weights_1 = (item_weights_1 - mean1) / std1
        item_weights_2 = (item_weights_2 - mean2) / std2
        item_weights_3 = (item_weights_3 - mean3) / std3
        item_weights_4 = (item_weights_4 - mean4) / std4

        item_weights = (
            item_weights_1 * self.weights["knn"]
            + item_weights_2 * self.weights["rp3"]
            + item_weights_3 * self.weights["ials"]
            + item_weights_4 * self.weights["top_pop"]
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

    rec = Hybrid(URM_all)
    best_w = {'knn_w': 0.5776327764428464, 'rp3_w': 0.38152295844133655, 'ials_w': 0.9206490029474442, 'top_pop_w': 0.18506014340806887}
    rec.fit(**best_w)

    create_submission(rec)


