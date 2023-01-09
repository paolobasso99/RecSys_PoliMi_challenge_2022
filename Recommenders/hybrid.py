from Recommenders.BaseRecommender import BaseRecommender

class Hybrid(BaseRecommender):
    def __init__(self, URM_train, recommenders):
        super().__init__(URM_train)
        self.recommenders = recommenders

    def fit(recommenders_weights):
        

    