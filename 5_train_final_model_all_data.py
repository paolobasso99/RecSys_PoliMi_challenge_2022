import os
from pathlib import Path
from skopt.space import Real
import scipy.sparse as sps

from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.MatrixFactorization.IALSRecommenderImplicit import (
    IALSRecommenderImplicit,
)
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.EASE_R.EASE_R_Recommender import (
    EASE_R_Recommender,
)
from Recommenders.Hybrid.HybridLinearImpressions import HybridLinearImpressions
from evaluation.evaluator import EvaluatorHoldout
from data_manager import DatasetLoader, DatasetSplitter, URMGenerator, UIMGenerator
from utils.load_best_hyperparameters import load_best_hyperparameters
from utils.create_submission import create_submission

base_recommenders = {
    "KNN": (ItemKNNCFRecommender, Path("result_experiments/KNN")),
    "RP3beta": (RP3betaRecommender, Path("result_experiments/RP3beta")),
    "iALS": (IALSRecommenderImplicit, Path("result_experiments/iALS")),
    "EASE_R": (EASE_R_Recommender, Path("result_experiments/EASE_R")),
    "SLIM": (SLIMElasticNetRecommender, Path("result_experiments/SLIM")),
}

hyperparameters_range_dictionary = {
    "alpha": Real(-4, 1),
    "beta": Real(-4, 4)
}

dataset_loader = DatasetLoader()
dataset_splitter = DatasetSplitter(dataset_loader)
dataset_train, dataset_val = dataset_splitter.load_interactions_train_val()
URM_generator = URMGenerator(dataset_train, dataset_val)
URM_train, URM_val = URM_generator.generate_implicit_URM()
evaluator = EvaluatorHoldout(URM_val, cutoff_list=[10])
impressions_train, impressions_val = dataset_splitter.load_impressions_train_val()
UIM_generator = UIMGenerator(impressions_train, impressions_val, URM_train.shape[0], URM_train.shape[1])
UIM_train, UIM_val = UIM_generator.generate()

best_weights = load_best_hyperparameters(Path("result_experiments/Hybrid"))

loaded_recommenders = {}
for recommender_id, (recommender_class, folder) in base_recommenders.items():
    URM_train_file = folder / "tuned_URM/URM_train.npz"
    if URM_train_file.exists():
        recommender_URM_train = sps.load_npz(URM_train_file)
        recommender_URM_val = sps.load_npz(folder / "tuned_URM/URM_val.npz")
        recommender_obj = recommender_class(recommender_URM_train + recommender_URM_val)
        recommender_obj.load_model(
            str(folder / "tuned_URM"),
            (recommender_class.RECOMMENDER_NAME + "_best_model_trained_on_everything.zip"),
        )
    else:
        print(f"WARNING: Using implicit URM for {recommender_id}")
        recommender_obj = recommender_class(URM_train + URM_val)
        recommender_obj.load_model(
            str(folder),
            (recommender_class.RECOMMENDER_NAME + "_best_model_trained_on_everything.zip"),
        )

    loaded_recommenders[recommender_id] = recommender_obj

output_folder_path = "result_experiments/Hybrid_impressions/"
recommender_class = HybridLinearImpressions
n_cases = 100
n_random_starts = int(n_cases * 0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

best_hyperparameters = load_best_hyperparameters(Path("result_experiments/Hybrid_impressions"))
recommender = HybridLinearImpressions(URM_train + URM_val, UIM_train + UIM_val, loaded_recommenders)
params = {**best_hyperparameters, **best_weights}
recommender.fit(**params)

create_submission(recommender)