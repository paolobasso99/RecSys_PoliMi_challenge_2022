import os
from pathlib import Path

import pandas as pd


class DatasetLoader(object):
    def __init__(self, raw_data_dir: Path = Path("data"), processed_data_dir: Path = Path("data/processed")) -> None:
        if not os.path.exists(raw_data_dir):
            raise FileNotFoundError(f"The path {str(raw_data_dir)} does not exists.")

        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir

        if not os.path.exists(self.processed_data_dir):
            os.mkdir(processed_data_dir)

        self.interactions_file_path = self.processed_data_dir / "interactions.csv"

    def load_interactions(self) -> pd.DataFrame:
        if not os.path.isfile(self.interactions_file_path):
            print("Interactions file does not exists, generating from raw data...")
            dataset = self.generate_dataset_from_raw_data()
            
            print("Saving created interactions dataset...")
            dataset.to_csv(self.interactions_file_path)
        else:
            print("Loading interactions dataset from file...")
            dataset = pd.read_csv(
                self.interactions_file_path,
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

        return dataset

    def generate_dataset_from_raw_data(self) -> pd.DataFrame:
        # Load raw data
        print("Loading raw data...")
        item_type = pd.read_csv(
            self.raw_data_dir / "data_ICM_type.csv",
            usecols=["item_id", "feature_id"],
            dtype={0: "Int64", 1: "Int64"},
        ).set_index("item_id")
        item_type = item_type.rename(
            columns={
                "item_id": "item_id",
                "feature_id": "type_id",
            }
        )
        item_length = pd.read_csv(
            self.raw_data_dir / "data_ICM_length.csv",
            usecols=["item_id", "data"],
            dtype={0: "Int64", 1: "Int64"},
        ).set_index("item_id")
        item_length = item_length.rename(columns={"data": "length"})
        interactions = pd.read_csv(
            self.raw_data_dir / "interactions_and_impressions.csv",
            dtype={0: "Int64", 1: "Int64", 2: str, 3: "Int64"},
        )
        interactions = interactions.rename(
            columns={
                "UserID": "user_id",
                "ItemID": "item_id",
                "Data": "data",
                "Impressions": "impressions",
            }
        )

        # Process raw data
        print("Processing raw data...")
        views = interactions[interactions["data"] == 0].drop(
            ["data", "impressions"], axis=1
        )
        details = interactions[interactions["data"] == 1].drop(
            ["data", "impressions"], axis=1
        )

        views["views_count"] = 1
        views_count = views.groupby(["user_id", "item_id"], as_index=False)[
            "views_count"
        ].sum()
        details["details_count"] = 1
        details_count = details.groupby(["user_id", "item_id"], as_index=False)[
            "details_count"
        ].sum()

        view_details_count = views_count.set_index(["user_id", "item_id"]).join(
            details_count.set_index(["user_id", "item_id"]),
            on=["user_id", "item_id"],
            how="outer",
        )
        view_details_count.loc[
            view_details_count["details_count"].isna(), "details_count"
        ] = 0
        view_details_count.loc[
            view_details_count["views_count"].isna(), "views_count"
        ] = 0
        view_details_count = view_details_count.join(item_length, on="item_id")
        view_details_count = view_details_count.join(item_type, on="item_id")
        view_details_count = view_details_count.reset_index()

        # Fill missing type_id and length
        view_details_count.loc[view_details_count["type_id"].isna(), "type_id"] = 1
        view_details_count.loc[view_details_count["length"].isna(), "length"] = 1

        return view_details_count
