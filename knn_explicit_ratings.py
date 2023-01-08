import os
import random

import numpy as np

from data_manager import DatasetLoader, DatasetSplitter, URMGenerator
from evaluation.evaluator import EvaluatorHoldout
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from skopt.space import Integer, Categorical
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from Recommenders.DataIO import DataIO

# Random seed for reproducibility
SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

dataset_loader = DatasetLoader()
dataset_splitter = DatasetSplitter(dataset_loader)
dataset_train, dataset_val = dataset_splitter.load_train_val()
URM_generator = URMGenerator(dataset_train, dataset_val)
URM_train, URM_val = URM_generator.generate_explicit_URM(log_base=4, views_weight=1.0, details_weight=0.8)
URM_all = URM_train + URM_val
evaluator = EvaluatorHoldout(URM_val, cutoff_list=[10])

output_folder_path = "result_experiments/ItemKNNCFRecommender/"
recommender_class = ItemKNNCFRecommender
n_cases = 30
n_random_starts = int(n_cases * 0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Define hyperparameters
hyperparameters_range_dictionary = {
    "topK": Integer(5, 2000),
    "shrink": Integer(0, 500),
    "similarity": Categorical(["cosine"]),
    "normalize": Categorical([True]),
}

hyperparameter_search = SearchBayesianSkopt(
    recommender_class,
    evaluator_validation=evaluator,
)

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[
        URM_train,
    ],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={},
)

hyperparameter_search.search(
    recommender_input_args,
    hyperparameter_search_space=hyperparameters_range_dictionary,
    n_cases=n_cases,
    n_random_starts=n_random_starts,
    save_model="best",
    output_folder_path=output_folder_path,  # Where to save the results
    output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
    metric_to_optimize=metric_to_optimize,
    cutoff_to_optimize=cutoff_to_optimize,
)

data_loader = DataIO(folder_path=output_folder_path)
search_metadata = data_loader.load_data(
    recommender_class.RECOMMENDER_NAME + "_metadata.zip"
)

result_on_validation_df = search_metadata["result_on_validation_df"]
print(result_on_validation_df)
best_hyperparameters = search_metadata["hyperparameters_best"]
print(best_hyperparameters)