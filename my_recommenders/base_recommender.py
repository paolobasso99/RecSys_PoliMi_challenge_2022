import numpy as np
import scipy.sparse as sps

class BaseRecommender:
    def __init__(self, URM: sps.csr_matrix, ICM, exclude_seen=True):
        if not sps.isspmatrix_csr(URM):
            raise TypeError(f"We expected a CSR matrix, we got {type(URM)}")
        self.URM = URM.copy()
        self.ICM = ICM.copy()
        self.predicted_URM = None
        self.exclude_seen = exclude_seen
        self.recommendations = None

    def fit(self):
        raise NotImplementedError()

    def recommend(self, user_id, at=10):
        predicted_ratings = self.compute_predicted_ratings(user_id)

        if self.exclude_seen:
            predicted_ratings = self.__filter_seen(user_id, predicted_ratings)

        # top k indices in sparse order
        ind = np.argpartition(predicted_ratings, -at)[-at:]
        # support needed to correctly index
        f = np.flip(np.argsort(predicted_ratings[ind]))
        return ind[f]

    def compute_predicted_ratings(self, user_id):
        return self.predicted_URM[user_id].toarray().ravel()

    def __filter_seen(self, user_id, predicted_ratings):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        predicted_ratings[user_profile] = -np.inf

        return predicted_ratings

    def compute_predicted_ratings_top_k(self, user_id, k):
        predicted_ratings = self.compute_predicted_ratings(user_id)

        if self.exclude_seen:
            predicted_ratings = self.__filter_seen(user_id, predicted_ratings)

        # top k indices in sparse order
        mask = np.argpartition(predicted_ratings, -k)[-k:]

        return predicted_ratings[mask], mask


class BaseMatrixFactorizationRecommender(BaseRecommender):
    def __init__(self, URM: sps.csr_matrix, ICM, exclude_seen=True):
        super().__init__(URM, ICM, exclude_seen)
        self.user_factors = None
        self.item_factors = None

    def compute_predicted_ratings(self, user_id):
        return np.dot(self.user_factors[user_id], self.item_factors.T)

    def fit(self):
        raise NotImplementedError()