import pandas as pd

from Recommenders.BaseRecommender import BaseRecommender
from data_manager.URM_generator import URMGenerator

class DatasetRecommender(BaseRecommender):
    def __init__(self, base_recommender_class, df_train: pd.DataFrame) -> None:
        self.base_recommender_class = base_recommender_class
        self.URMGenerator = URMGenerator(df_train, pd.DataFrame())

    def fit(self, **base_recommender_fit_args):
        URM_train, _ = self.URMGenerator.generate_explicit_URM()
        base_recommender = self.base_recommender_class(URM_train)
        return base_recommender.fit(base_recommender_fit_args)