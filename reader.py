from pathlib import Path

import pandas as pd

_DEFAULT_BASE_DIR = Path("Data")


class Dataset(object):
    def __init__(self, basedir: Path = _DEFAULT_BASE_DIR) -> None:
        self.basedir = basedir

        self.load_item_type_df()
        self.load_item_lenght_df()
        self.load_interactions_df()
        self.load_views_details_df()

    def load_item_type_df(self) -> pd.DataFrame:
        self.item_type_df = pd.read_csv(
            self.basedir / "data_ICM_type.csv",
            usecols=["item_id", "feature_id"],
            dtype={0: int, 1: int},
        )
        return self.item_type_df

    def load_item_lenght_df(self) -> pd.DataFrame:
        self.item_length_df = pd.read_csv(
            self.basedir / "data_ICM_length.csv",
            usecols=["item_id", "data"],
            dtype={0: int, 1: int},
        )
        return self.item_length_df

    def load_interactions_df(self) -> pd.DataFrame:
        interactions_df = pd.read_csv(
            self.basedir / "interactions_and_impressions.csv",
            dtype={0: int, 1: int, 2: str, 3: int},
        )
        self.interactions_df = interactions_df.rename(
            columns={
                "UserID": "user_id",
                "ItemID": "item_id",
                "Data": "data",
                "Impressions": "impressions",
            }
        )
        return self.interactions_df

    def load_views_details_df(self) -> pd.DataFrame:
        if self.interactions_df is None:
            self.interactions_df = self.load_interactions_df()
        self.views = self.interactions_df[self.interactions_df["data"] == 0].drop(
            ["data", "impressions"], axis=1
        )
        self.details = self.interactions_df[self.interactions_df["data"] == 1].drop(
            ["data", "impressions"], axis=1
        )
        return self.views, self.details

    

if __name__ == "__main__":
    ds = Dataset()