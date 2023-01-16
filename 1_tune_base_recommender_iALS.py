from hyperparameter_tuning.tune_base_recommender import tune_base_recommender
from Recommenders.MatrixFactorization.IALSRecommenderImplicit import IALSRecommenderImplicit

tune_base_recommender(
    recommender_class=IALSRecommenderImplicit,
    n=100,
    output_folder="iALS",
    save_trained_on_all=True
)