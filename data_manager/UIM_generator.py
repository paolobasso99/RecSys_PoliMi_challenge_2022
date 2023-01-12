from typing import Tuple

import pandas as pd
import scipy.sparse as sps


class UIMGenerator(object):
    def __init__(
        self, impressions_train: pd.DataFrame, impressions_val: pd.DataFrame, n_users: int, n_items: int
    ) -> None:
        self.impressions_train = impressions_train.copy()
        self.impressions_val = impressions_val.copy()
        self.n_users = n_users
        self.n_items = n_items

    def _generate_split(
        self, dataset: pd.DataFrame
    ) -> sps.coo_matrix:
        UIM = sps.coo_matrix(
            (
                dataset["impressions_count"].values.astype("float64"),
                (dataset["user_id"].values, dataset["item_id"].values),
            ),
            shape=(self.n_users, self.n_items),
        )
        return UIM.tocsr()

    def generate(
        self
    ) -> Tuple[sps.coo_matrix, sps.coo_matrix]:
        print("Generating UIM...")

        UIM_train = self._generate_split(
            self.impressions_train
        )
        UIM_val = self._generate_split(
            self.impressions_val
        )

        return UIM_train, UIM_val
