import os
from data_manager import DatasetLoader, DatasetSplitter, URMGenerator
from evaluation.evaluator import EvaluatorHoldout
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from skopt.space import Real, Integer
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from Recommenders.DataIO import DataIO

if __name__ == "__main__":
    dataset_loader = DatasetLoader()
    dataset_splitter = DatasetSplitter(dataset_loader)
    dataset_train, dataset_val = dataset_splitter.load_interactions_train_val()
    URM_generator = URMGenerator(dataset_train, dataset_val)
    URM_train, URM_val = URM_generator.generate_explicit_URM(log_base=84, views_weight=93, details_weight=71)
    URM_all = URM_train + URM_val

    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[10])
    
    output_folder_path = "result_experiments/SLIMElasticNet2/"
    recommender_class = SLIMElasticNetRecommender
    n_cases = 100
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Define hyperparameters
    hyperparameters_range_dictionary = {
        "topK": Integer(500, 2000),
        "l1_ratio": Real(low=1e-4, high=0.1, prior="log-uniform"),
        "alpha": Real(low=1e-4, high=0.1, prior="uniform"),
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

    if True:
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

    best_hyperparameters = search_metadata["hyperparameters_best"]
    print("best_param", best_hyperparameters)
    
    recommender = recommender_class(URM_train + URM_val)
    recommender.fit(**best_hyperparameters)
    recommender.save_model(
        folder_path=output_folder_path,
        file_name=recommender_class.RECOMMENDER_NAME
        + "_best_model_trained_on_everything.zip",
    )