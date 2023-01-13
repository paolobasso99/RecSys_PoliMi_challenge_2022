import os
from data_manager import DatasetLoader, DatasetSplitter, URMGenerator
from evaluation.evaluator import EvaluatorHoldout
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_SVDpp_Cython
from skopt.space import Real, Integer, Categorical
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from Recommenders.DataIO import DataIO

if __name__ == "__main__":
    dataset_loader = DatasetLoader()
    dataset_splitter = DatasetSplitter(dataset_loader)
    dataset_train, dataset_val = dataset_splitter.load_interactions_train_val()
    URM_generator = URMGenerator(dataset_train, dataset_val)
    URM_train, URM_val = URM_generator.generate_explicit_URM(
        log_base=4, views_weight=1, details_weight=0.8
    )
    URM_all = URM_train + URM_val

    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[10])

    output_folder_path = "result_experiments/MF_SVDpp/"
    recommender_class = MatrixFactorization_SVDpp_Cython
    n_cases = 100
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Define hyperparameters
    hyperparameters_range_dictionary = {
        "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
        "epochs": Categorical([500]),
        "use_bias": Categorical([True, False]),
        "batch_size": Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
        "num_factors": Integer(1, 200),
        "item_reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
        "user_reg": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
        "learning_rate": Real(low = 1e-4, high = 1e-1, prior = 'log-uniform'),
        "negative_interactions_quota": Real(low = 0.0, high = 0.5, prior = 'uniform'),
        "dropout_quota": Real(low = 0.01, high = 0.7, prior = 'uniform')
    }

    hyperparameter_search = SearchBayesianSkopt(
        recommender_class,
        evaluator_validation=evaluator,
    )

    earlystopping_keywargs = {
        "validation_every_n": 5,
        "stop_on_validation": True,
        "evaluator_object": evaluator,
        "lower_validations_allowed": 5,
        "validation_metric": metric_to_optimize,
    }

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS=earlystopping_keywargs,
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
