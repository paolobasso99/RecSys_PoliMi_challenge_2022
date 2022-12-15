from Recommenders.BaseRecommender import BaseRecommender

class Hybrid(BaseRecommender):
    def __init__(self, URM_train):
        super().__init__(URM_train)
        self.recommenders = {}
        self.weights = {}

    