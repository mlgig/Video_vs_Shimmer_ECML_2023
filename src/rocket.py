import argparse
import configparser
import os
import logging
from pathlib import Path
import time

from configobj import ConfigObj
from sklearn import metrics
import numpy as np
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import sktime
import pandas as pd

from utils.program_stats import timeit
from utils.sklearn_utils import report_average, plot_confusion_matrix
from utils.util_functions import create_directory_if_not_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILE_NAME_X = '{}_{}_X'
FILE_NAME_Y = '{}_{}_Y'
FILE_NAME_PID = '{}_{}_pid'


def read_dataset(path, data_type):
    """
    This function is used to read the data stored in sktime format. Here dat
    :param path: path to the dataset
    :param data_type: type of data. The values is "default" here
    :return: loaded train, test and val data in the pandas dataframe
    """
    x_train, y_train = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                  FILE_NAME_X.format("TRAIN", data_type) + ".ts"))

    logger.info("Training data shape {} {} {}".format(x_train.shape, len(x_train.iloc[0, 0]), y_train.shape))
    x_test, y_test = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                FILE_NAME_X.format("TEST", data_type) + ".ts"))
    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))

    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))
    test_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TEST", data_type) + ".npy"), allow_pickle=True)
    train_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TRAIN", data_type) + ".npy"), allow_pickle=True)

    try:
        x_val, y_val = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                  FILE_NAME_X.format("VAL", data_type) + ".ts"))
        logger.info("Validation data shape: {} {}".format(x_val.shape, y_val.shape))
    except (sktime.utils.data_io.TsFileParseException, FileNotFoundError):
        logger.info("Validation data is empty:")
        x_val, y_val = None, None

    return x_train, y_train, x_test, y_test, x_val, y_val, train_pid, test_pid


class RocketTransformerClassifier:
    """
    Class to perform the classification using ROCKET as classifier for time series.
    Sktime library is used to implement the ROCKET. It expects the data in the sktime formatted.
    """

    def __init__(self, exercise):
        """
        :param exercise: exercise name MP or Rowing
        """
        self.exercise = exercise
        # classifier mapping used to store the trained model and transformed method
        self.classifiers_mapping = {}

    @timeit
    def fit_rocket(self, x_train, y_train, train_pid, kernels=10000):
        """
        :param x_train: training data in the sktime format
        :param y_train: target labels
        :param train_pid: info about each record in the training data. Basically it provides participant id and repetition
        number for a particular exercise type.This can be used to create further splits of the data without data leakage.
        :param kernels: number of kernels to be used for classification for ROCKET.
        :return:
        """
        # Normalize flag is turned on for IMU data and turned off for time series obtained from video data.
        rocket = Rocket(num_kernels=kernels, normalise=False)
        rocket.fit(x_train)
        x_training_transform = rocket.transform(x_train)
        self.classifiers_mapping["transformer"] = rocket

        # We can also use Ridge classifier instead of RidgeClassifierCV. The tuned alpha found for MP is 0.1 for
        # MP for video case
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier.fit(x_training_transform, y_train)

        # To calculate probabilities values for training data
        # Training Predictions
        # predictions = classifier.predict(x_training_transform)
        # d = classifier.decision_function(x_training_transform)
        # probs = np.exp(d) / np.sum(np.exp(d), axis=1).reshape(-1, 1)
        self.classifiers_mapping["classifier"] = classifier

    @timeit
    def predict_rocket(self, x_test, y_test, test_pid):
        """

        :param x_test: test data in the sktime format
        :param y_test: test labels
        :param test_pid: information about the participants basically repetition number, exercise type
        :return:
        """
        rocket = self.classifiers_mapping["transformer"]
        classifier = self.classifiers_mapping["classifier"]
        x_test_transform = rocket.transform(x_test)

        # Test Predictions
        # Used to calculate the probabilities for each instance
        predictions = classifier.predict(x_test_transform)
        # d = self.classifiers_mapping["classifier"].decision_function(x_test_transform)
        # probs = np.exp(d) / np.sum(np.exp(d), axis=1).reshape(-1, 1)

        # Confusion Matrix and classification report
        labels = list(np.sort(np.unique(y_test)))
        confusion_matrix = metrics.confusion_matrix(y_test, predictions)
        classification_report = metrics.classification_report(y_test, predictions)
        logger.info("-----------------------------------------------")
        logger.info("Metrics on testing data")
        logger.info("Accuracy {}".format(metrics.accuracy_score(y_test, predictions)))
        logger.info("\n Confusion Matrix: \n {}".format(confusion_matrix))
        logger.info("\n Classification report: \n{}".format(classification_report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Use a config file to pass the arguments
    parser.add_argument("--rocket_config", required=True, help="path of the config file")
    args = parser.parse_args()
    rocket_config = ConfigObj(args.rocket_config)

    base_path = os.path.dirname(os.getcwd())  # base path of the project where the file is running from
    input_data_path = rocket_config["INPUT_DATA_PATH"]
    exercise = rocket_config["EXERCISE"]  # exercise basically MP or Rowing
    output_path = rocket_config["OUTPUT_PATH"]
    data_type = rocket_config["DATA_TYPE"]  # here the value is default

    output_results_path = os.path.join(output_path, "Rocket")
    create_directory_if_not_exists(output_results_path)

    classification_report_list = []
    logger.info("----------------------------------------------------")
    logger.info("Fitting Rocket")
    input_path_combined = os.path.join(input_data_path, exercise)
    if not os.path.exists(input_path_combined):
        logger.info("Path does not exist")
        exit(1)
    x_train, y_train, x_test, y_test, x_val, y_val, train_pid, test_pid = read_dataset(input_path_combined, data_type)
    ts = time.time()
    rocket_classifier = RocketTransformerClassifier(exercise)
    rocket_classifier.fit_rocket(x_train, y_train, train_pid)
    rocket_classifier.predict_rocket(x_test, y_test, test_pid)
    te = time.time()
    total_time = (te - ts)
    logger.info('Total time preprocessing: {} seconds'.format(total_time))

