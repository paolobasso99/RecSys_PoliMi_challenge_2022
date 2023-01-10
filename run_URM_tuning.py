import scipy.sparse as sps
from Recommenders.GraphBased.RP3betaRecommender import (
    RP3betaRecommender,
)
from utils.URM_tuner import URM_tuner
from retrain_with_tuned_URM import retrain_with_tuned_URM
from pathlib import Path
from data_manager import DatasetLoader, DatasetSplitter, URMGenerator

if __name__ == "__main__":
    recommender_class = RP3betaRecommender
    model_folder = Path("result_experiments/RP3beta/")
    N = 100
    
    if True:
        log_base, views_weight, details_weight = URM_tuner(
            N, recommender_class, model_folder
        )
    else:
        log_base = 12.62329790627502
        views_weight = 82.89881321660107
        details_weight = 27.811514733648256

    dataset_loader = DatasetLoader()
    dataset_splitter = DatasetSplitter(dataset_loader)
    dataset_train, dataset_val = dataset_splitter.load_train_val()
    URM_generator = URMGenerator(dataset_train, dataset_val)
    URM_train, URM_val = URM_generator.generate_explicit_URM(
        log_base=log_base, views_weight=views_weight, details_weight=details_weight
    )

    retrain_with_tuned_URM(recommender_class, model_folder, URM_train, URM_val)

    sps.save_npz(model_folder / "tuned_URM/URM_train.npz", URM_train)
    sps.save_npz(model_folder / "tuned_URM/URM_val.npz", URM_val)
