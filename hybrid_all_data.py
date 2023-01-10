import numpy as np

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

from utils.create_submission import create_submission


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

    rec = Hybrid(URM_all)
    best_w = {'knn_w': 0.8417663448977378, 'rp3_w': 0.48402963897293105, 'ials_w': 0.8927831950366801, 'top_pop_w': 0.24393946868506233, 'ease_r_w': 0.4248184758985911}
    rec.fit(**best_w)

    create_submission(rec)


