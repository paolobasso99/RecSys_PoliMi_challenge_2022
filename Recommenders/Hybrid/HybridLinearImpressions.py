import numpy as np
import scipy.sparse as sps

from Recommenders.BaseRecommender import BaseRecommender

class HybridLinearImpressions(BaseRecommender):
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

    def fit(self, alpha, beta, KNN, RP3beta, iALS, EASE_R, SLIM):
        self.weights = {
            "KNN": KNN,
            "RP3beta": RP3beta,
            "iALS": iALS,
            "EASE_R": EASE_R,
            "SLIM": SLIM
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