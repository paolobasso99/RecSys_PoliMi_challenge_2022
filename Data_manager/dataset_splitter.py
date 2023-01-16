import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from data_manager.dataset_loader import DatasetLoader


class DatasetSplitter(object):
    def __init__(
        self,
        dataset_loader: DatasetLoader,
        splitted_data_dir: Path = Path("data/processed/splitted"),
    ) -> None:
        self.dataset_loader = dataset_loader

        if not os.path.exists(splitted_data_dir):
            os.mkdir(splitted_data_dir)

        self.splitted_data_dir = splitted_data_dir
        self.interactions_train_path = self.splitted_data_dir / "interactions_train.csv"
        self.interactions_val_path = self.splitted_data_dir / "interactions_val.csv"
        self.impressions_train_path = self.splitted_data_dir / "impressions_train.csv"
        self.impressions_val_path = self.splitted_data_dir / "impressions_val.csv"

    def generate_train_val(self, test_size=0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
        interactions = self.dataset_loader.load_interactions()

        interactions_train, interactions_val = train_test_split(
            interactions,
            test_size=test_size,
            shuffle=True,
            stratify=interactions["user_id"],
        )

        return interactions_train, interactions_val

    def load_impressions_train_val(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not os.path.isfile(self.impressions_train_path) or not os.path.isfile(
            self.impressions_val_path
        ):
        else:
            dataset_train = pd.read_csv(self.impressions_train_path)
            dataset_val = pd.read_csv(self.impressions_val_path)
            return dataset_train, dataset_val

    def load_interactions_train_val(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not os.path.isfile(self.interactions_train_path) or not os.path.isfile(
            self.interactions_val_path
        ):
            print("Splits do not exists, generating...")
            dataset_train, dataset_val = self.generate_train_val()

            print("Saving generated splits...")
            dataset_train.to_csv(self.interactions_train_path)
            dataset_val.to_csv(self.interactions_val_path)
        else:
            print("Loading previously generated splits...")
            dataset_train = pd.read_csv(
                self.interactions_train_path,
                usecols=[
                    "user_id",
                    "item_id",
                    "views_count",
                    "details_count",
                    "length",
                    "type_id",
                ],
                dtype={
                    0: "Int64",
                    1: "Int64",
                    2: "Int64",
                    3: "Int64",
                    4: "Int64",
                    5: "Int64",
                },
            )
            dataset_val = pd.read_csv(
                self.interactions_val_path,
                usecols=[
                    "user_id",
                    "item_id",
                    "views_count",
                    "details_count",
                    "length",
                    "type_id",
                ],
                dtype={
                    0: "Int64",
                    1: "Int64",
                    2: "Int64",
                    3: "Int64",
                    4: "Int64",
                    5: "Int64",
                },
            )

        return dataset_train, dataset_val
