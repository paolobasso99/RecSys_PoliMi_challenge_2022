from hyperparameter_tuning.tune_base_recommender import tune_base_recommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

tune_base_recommender(
    recommender_class=SLIMElasticNetRecommender,
    n=100,
    output_folder="SLIM",
    save_trained_on_all=False
)