"""
Source: https://github.com/fulcus/recommender-systems-challenge/blob/master/Recommenders/MatrixFactorization/IALSRecommender_implicit.py
"""

import implicit
import numpy as np

from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender


class IALSRecommenderImplicit(BaseMatrixFactorizationRecommender):
    """
    ALS implemented with implicit following guideline of
    https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
    IDEA:
    Recomputing x_{u} and y_i can be done with Stochastic Gradient Descent, but this is a non-convex optimization problem.
    We can convert it into a set of quadratic problems, by keeping either x_u or y_i fixed while optimizing the other.
    In that case, we can iteratively solve x and y by alternating between them until the algorithm converges.
    This is Alternating Least Squares.
    """

    RECOMMENDER_NAME = "IALSRecommenderImplicit"

    def __init__(self, URM_train, verbose=True):
        super(IALSRecommenderImplicit, self).__init__(URM_train, verbose=verbose)

    def fit(self, num_factors=50, reg=0.001847510119137634, epochs=30, alpha=2, num_threads=2):
        self.num_factors = num_factors
        self.reg = reg
        self.epochs = epochs

        sparse_item_user = self.URM_train

        # Initialize the als model and fit it using the sparse item-user matrix
        model = implicit.als.AlternatingLeastSquares(factors=self.num_factors, regularization=self.reg,
                                                     iterations=self.epochs, num_threads=num_threads)

        # Calculate the confidence by multiplying it by our alpha value.
        data_conf = (sparse_item_user * alpha).astype('double')

        # Fit the model
        model.fit(data_conf)

        # Get the user and item vectors from our trained model
        self.USER_factors = model.user_factors
        self.ITEM_factors = model.item_factors
