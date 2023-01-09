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
        self.train_file_path = self.splitted_data_dir / "train.csv"
        self.val_file_path = self.splitted_data_dir / "val.csv"

    def generate_train_val(self, test_size=0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataset = self.dataset_loader.load_dataset()

        dataset_train, dataset_val = train_test_split(
            dataset,
            test_size=test_size,
            shuffle=True,
            stratify=dataset["user_id"],
        )

        return dataset_train, dataset_val

    def load_train_val(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not os.path.isfile(self.train_file_path) or not os.path.isfile(
            self.val_file_path
        ):
            print("Splits do not exists, generating...")
            dataset_train, dataset_val = self.generate_train_val()

            print("Saving generated splits...")
            dataset_train.to_csv(self.train_file_path)
            dataset_val.to_csv(self.val_file_path)
        else:
            print("Loading previusly generated splits...")
            dataset_train = pd.read_csv(
                self.train_file_path,
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
                self.val_file_path,
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
