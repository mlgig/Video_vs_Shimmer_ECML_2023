import argparse
import configparser
import os
import logging
import sys
from pathlib import Path
import time
import pickle
import traceback

import pandas as pd
import numpy as np

from src.program_stats import timeit
from src.signal_features import get_features
from src.utils import split_pids, apply_butter_worth_filter, resample_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sensor_body_parts_mapping = {"B8E3": "Right Wrist", "S57D0": "Left Wrist", "B8C7": "Right Arm", "S548D": "Left Arm",
                             "S9276": "Back"}
sensor_type_signals = ["_Gyro_", "_Accel_", "_Mag_", "pitch", "roll", "yaw"]
exercise_types_mapping = {"MP": ["A", "Arch", "N", "R"], "Rowing": ["A", "Ext", "N", "R", "RB"]}
order_features = ["no_mean_crossings", "5th_percentile", "25th_percentile", "75th_percentile", "median", "mean", "std",
                  "var", "rms", "min_val", "max_val", "range", "kurtosis", "skewness", "var_a", "var_d",
                  "fractal_dimension", "energy", "pid", "exercise_type"]
SEED_VALUES = [103007, 1899797, 191099]
pid_list = []

order_cols = ['B8C7_Accel_Magnitude', 'B8C7_Gyro_Magnitude', 'B8C7_pitch', 'B8C7_roll', 'B8C7_yaw',
              'B8E3_Accel_Magnitude',
              'B8E3_Gyro_Magnitude', 'B8E3_Accel_X', 'B8E3_Accel_Y', 'B8E3_Accel_Z', 'B8E3_Gyro_X', 'B8E3_Gyro_Y',
              'B8E3_Gyro_Z',
              'B8E3_Mag_X', 'B8E3_Mag_Y', 'B8E3_Mag_Z', 'B8E3_pitch', 'B8E3_roll', 'B8E3_yaw', 'S548D_Accel_Magnitude',
              'S548D_Gyro_Magnitude', 'S548D_Accel_X', 'S548D_Accel_Y', 'S548D_Accel_Z', 'S548D_Gyro_X', 'S548D_Gyro_Y',
              'S548D_Gyro_Z', 'S548D_Mag_X', 'S548D_Mag_Y', 'S548D_Mag_Z', 'S548D_pitch', 'S548D_roll', 'S548D_yaw',
              'S57D0_Accel_X',
              'S57D0_Accel_Y', 'S57D0_Accel_Z', 'S57D0_Gyro_X', 'S57D0_Gyro_Y', 'S57D0_Gyro_Z', 'S57D0_Mag_X',
              'S57D0_Mag_Y',
              'S57D0_Mag_Z', 'S57D0_Accel_Magnitude', 'S57D0_Gyro_Magnitude', 'S57D0_pitch', 'S57D0_roll', 'S57D0_yaw',
              'S9276_Accel_X', 'S9276_Accel_Y', 'S9276_Accel_Z', 'S9276_Accel_Magnitude', 'S9276_Gyro_Magnitude',
              'S9276_Gyro_X',
              'S9276_Gyro_Y', 'S9276_Gyro_Z', 'S9276_Mag_X', 'S9276_Mag_Y', 'S9276_Mag_Z', 'S9276_pitch', 'S9276_roll',
              'S9276_yaw',
              'B8C7_Accel_X', 'B8C7_Accel_Y', 'B8C7_Accel_Z', 'B8C7_Gyro_X', 'B8C7_Gyro_Y', 'B8C7_Gyro_Z', 'B8C7_Mag_X',
              'B8C7_Mag_Y', 'B8C7_Mag_Z']


def generate_manual_features(path_coordinates, coordinate_file):
    file_path = os.path.join(path_coordinates, coordinate_file)
    full_df = pd.read_csv(file_path)
    pid, label = coordinate_file[:-4].split("_")
    pid_list.append(pid)
    nonimp_cols = ["frame_number", "frame_peaks", "pid", "sample_id"]
    unique_sample_id_list = full_df["sample_id"].unique()
    segment_df_list = []

    for sample_id in unique_sample_id_list:
        segment_df = full_df[full_df["sample_id"] == sample_id].copy()
        segment_df.drop(nonimp_cols, axis=1, inplace=True)
        segment_df_list.append(segment_df)

    column_list = segment_df_list[0].columns.tolist()
    column_list.sort()
    # logger.info("Column order: {}, {}".format(column_list, len(column_list)))
    segment_df_list = [df[column_list] for df in segment_df_list]

    new_features = []
    filter_columns = []

    # for c in column_list:
    #     if "S57D0" in c or "B8E3" in c:
    #         filter_columns.append(c)

    # for c in column_list:
    #     if "Magnitude" in c or "pitch" in c or "roll" in c or "yaw" in c:
    #         continue
    #     filter_columns.append(c)
    if not filter_columns:
        filter_columns = column_list

    logger.info("Length of filter columns: {}".format(len(filter_columns)))
    for df in segment_df_list:
        features = []
        for c in filter_columns:
            """
            "B8E3": "Right Wrist", "S57D0": "Left Wrist", "B8C7": "Right Arm", "S548D": "Left Arm",
                             "S9276": "Back"
            """
            filter_signal = apply_butter_worth_filter(df[c])
            resampled_signal = resample_signal(filter_signal)
            # resampled_signal = (resampled_signal - resampled_signal.mean()) / resampled_signal.std()
            features += get_features(resampled_signal)
        features.append(pid)
        features.append(label)
        new_features.append(features)
    return new_features


def create_train_test_split(full_df, current_pids_list, seed_value, split_ratio=0.3):
    train_pids, test_pids = split_pids(current_pids_list, seed_value, split_ratio)
    logger.info("Total persons in training: {}".format(train_pids))
    logger.info("Total persons in test: {}".format(test_pids))

    train_df = full_df[full_df.iloc[:, -2].isin(train_pids)]
    test_df = full_df[full_df.iloc[:, -2].isin(test_pids)]
    # train_df.drop([1260, 1261], axis=1, inplace=True)
    # test_df.drop([1260, 1261], axis=1, inplace=True)
    logger.info("Full Data Shape: {}".format(full_df.shape))
    logger.info("Train Data Shape: {}".format(train_df.shape))
    logger.info("Test Data Shape: {}".format(test_df.shape))
    train_df.to_csv("{}/{}/train_{}.csv".format(output_path, seed_value, data_type), index=False)
    test_df.to_csv("{}/{}/test_{}.csv".format(output_path, seed_value, data_type), index=False)


@timeit
def main(path_coordinates):
    unique_coordinates_files = os.listdir(path_coordinates)
    ts = time.time()
    total_features = []
    count = 0
    for coordinate_file in unique_coordinates_files:
        logger.info("Running for {}".format(coordinate_file))
        try:
            if not coordinate_file.startswith('.'):
                features = generate_manual_features(path_coordinates, coordinate_file)
                total_features.extend(features)
                count += 1
            if count % 50 == 0:
                logger.info("Generated manual features for {} files:".format(count))
        except Exception as e:
            logger.info("Error in generating the coordinates for: {} {}".format(coordinate_file, str(e)))
            logger.info(traceback.format_exc())
    final_df = pd.DataFrame(total_features)
    final_df.to_csv("/tmp/final_df.csv", index=False)
    te = time.time()
    total_time = (te - ts)
    logger.info('Total time preprocessing: {} seconds'.format(total_time))
    return final_df


if __name__ == "__main__":
    data_type = "default_temppp"
    output_path = "/home/ashish/Results/Datasets/Shimmer/Rowing/Manual"
    seg_path_coordinates = "/home/ashish/Results/Datasets/Shimmer/Rowing/SegmentedCoordinates"
    final_df = main(seg_path_coordinates)
    # final_df = pd.read_csv("/tmp/final_df.csv")
    current_pids_list = np.unique(final_df.iloc[:, -2])
    for sv in SEED_VALUES:
        logger.info('Seed value: {}'.format(sv))
        create_train_test_split(final_df, current_pids_list, sv)

"""
default: 238 seconds

282

"""
