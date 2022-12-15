from typing import Tuple

import pandas as pd
import numpy as np
import scipy.sparse as sps


class URMGenerator(object):
    def __init__(
        self, dataset_training: pd.DataFrame, dataset_val: pd.DataFrame
    ) -> None:
        self.dataset_training = dataset_training.copy()
        self.dataset_val = dataset_val.copy()

        # Find n_users and n_items
        users_id = np.unique(np.concatenate((self.dataset_training["user_id"], self.dataset_val["user_id"])))
        items_id = np.unique(np.concatenate((self.dataset_training["item_id"], self.dataset_val["item_id"])))
        self.n_users = users_id.shape[0]
        self.n_items = items_id.shape[0]

    def _geneate_explicit_split_URM(self, dataset: pd.DataFrame, alpha = 0.1, details_weight = 0.1) -> sps.coo_matrix:
        dataset["views_ratings"] = dataset["views_count"] / (
            dataset["views_count"] + alpha * dataset["length"]
        )
        dataset["details_ratings"] = dataset["details_count"] / (
            dataset["details_count"] + alpha * dataset["length"]
        )
        dataset["combined_ratings"] = (
            (1-details_weight) * dataset["views_ratings"]
            + details_weight * dataset["details_ratings"]
        )

        URM = sps.coo_matrix(
            (
                dataset["combined_ratings"].values.astype('float64'),
                (dataset["user_id"].values, dataset["item_id"].values),
            ),
            shape=(self.n_users, self.n_items)
        )
        return URM

    def generate_explicit_URM(self, alpha = 0.1, details_weight = 0.1) -> Tuple[sps.coo_matrix, sps.coo_matrix]:
        print("Generating explicit URM...")
        if details_weight < 0 or details_weight > 1:
            raise ValueError("URM_generator (generate_URM): details_weight must be in [0,1]")

        URM_train = self._geneate_explicit_split_URM(self.dataset_training, alpha, details_weight)
        URM_val = self._geneate_explicit_split_URM(self.dataset_val, alpha, details_weight)

        return URM_train, URM_val

    def _geneate_impicit_split_URM(self, dataset: pd.DataFrame) -> sps.coo_matrix:
        dataset["interacted"] = 1

        URM = sps.coo_matrix(
            (
                dataset["interacted"].values,
                (dataset["user_id"].values, dataset["item_id"].values),
            ),
            shape=(self.n_users, self.n_items)
        )
        return URM

    def generate_implicit_URM(self) -> Tuple[sps.coo_matrix, sps.coo_matrix]:
        print("Generating implicit URM...")

        URM_train = self._geneate_impicit_split_URM(self.dataset_training)
        URM_val = self._geneate_impicit_split_URM(self.dataset_val)

        return URM_train, URM_val
