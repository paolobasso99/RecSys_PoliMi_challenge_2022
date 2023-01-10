from pathlib import Path

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.MatrixFactorization.IALSRecommenderImplicit import IALSRecommenderImplicit
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from utils.load_best_hyperparameters import load_best_hyperparameters
from data_manager import DatasetLoader, DatasetSplitter, URMGenerator

def retrain_with_tuned_URM(recommender_class: BaseRecommender, folder: Path, URM_train, URM_val):
    print(f"Retraining {recommender_class.RECOMMENDER_NAME} with tuned URM")
    best_hyperparameters = load_best_hyperparameters(folder)
    recommender = recommender_class(URM_train)
    print("Fit URM_train")
    recommender.fit(**best_hyperparameters)
    recommender.save_model(
        folder_path=str(folder / "tuned_URM") + "/",
        file_name=recommender_class.RECOMMENDER_NAME
        + "_best_model.zip",
    )

    recommender = recommender_class(URM_train + URM_val)
    print("Fit URM_all")
    recommender.fit(**best_hyperparameters)
    recommender.save_model(
        folder_path=str(folder / "tuned_URM") + "/",
        file_name=recommender_class.RECOMMENDER_NAME
        + "_best_model_trained_on_everything.zip",
    )

if __name__ == "__main__":
    URM_parameters = {
        "log_base": 81.12913520607813,
        "views_weight": 89.60912999234932,
        "details_weight": 31.800347497186387
    }
    to_train = {
        ItemKNNCFRecommender: Path("result_experiments/KNN"),
        #IALSRecommenderImplicit: Path("result_experiments/IALS"),
        RP3betaRecommender: Path("result_experiments/RP3beta"),
        EASE_R_Recommender: Path("result_experiments/EASE_R"),
    }


    dataset_loader = DatasetLoader()
    dataset_splitter = DatasetSplitter(dataset_loader)
    dataset_train, dataset_val = dataset_splitter.load_train_val()
    URM_generator = URMGenerator(dataset_train, dataset_val)
    URM_train, URM_val = URM_generator.generate_explicit_URM(**URM_parameters)

    for recommender_class, folder in to_train.items():
        retrain_with_tuned_URM(recommender_class, folder, URM_train, URM_val)