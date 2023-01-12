import scipy.sparse as sps
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from utils.URM_tuner import URM_tuner
from retrain_with_tuned_URM import retrain_with_tuned_URM
from pathlib import Path
from data_manager import DatasetLoader, DatasetSplitter, URMGenerator

if __name__ == "__main__":
    recommender_class = SLIMElasticNetRecommender
    model_folder = Path("result_experiments/SLIMElasticNet/")
    N = 100
    run_tuner = True
    
    dataset_loader = DatasetLoader()
    dataset_splitter = DatasetSplitter(dataset_loader)
    dataset_train, dataset_val = dataset_splitter.load_train_val()
    URM_generator = URMGenerator(dataset_train, dataset_val)
    
    if run_tuner:
        log_base, views_weight, details_weight = URM_tuner(
            N, recommender_class, model_folder
        )
    else:
        log_base = 81.12913520607813
        views_weight = 89.60912999234932
        details_weight = 31.800347497186387

    URM_train, URM_val = URM_generator.generate_explicit_URM(
        log_base=log_base, views_weight=views_weight, details_weight=details_weight
    )

    retrain_with_tuned_URM(recommender_class, model_folder, URM_train, URM_val)

    sps.save_npz(model_folder / "tuned_URM/URM_train.npz", URM_train)
    sps.save_npz(model_folder / "tuned_URM/URM_val.npz", URM_val)
