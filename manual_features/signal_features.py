import numpy as np
import pandas as pd
import math
from scipy.stats import kurtosis
from scipy.stats import skew, variation
import pywt


def calc_energy(series):
    energy = np.sum(np.abs(series) ** 2)
    return [energy / len(series)]


def calc_fractal_dimension(series):
    """
     Compute Petrosian Fractal Dimension of a time series
    :param series:
    :return:
    """
    diff = np.diff(series)
    # x[i] * x[i-1] for i in t0 -> tmax
    prod = diff[1:-1] * diff[0:-2]

    # Number of sign changes in derivative of the signal
    N_delta = np.sum(prod < 0)
    n = len(series)
    fd = np.log(n) / (np.log(n) + np.log(n / (n + 0.4 * N_delta)))
    return [fd]


def calculate_statistics(series):
    n5 = np.nanpercentile(series, 5)
    n25 = np.nanpercentile(series, 25)
    n75 = np.nanpercentile(series, 75)
    median = np.nanpercentile(series, 50)
    mean = np.nanmean(series)
    std = np.nanstd(series)
    var = np.nanvar(series)
    rms = np.nanmean(np.sqrt(series ** 2))
    min_val = np.min(series)
    max_val = np.max(series)
    rng = max_val - min_val
    krt = kurtosis(series)
    skw = skew(series)
    return [n5, n25, n75, median, mean, std, var, rms, min_val, max_val, rng, krt, skw]


def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_mean_crossings]


def var_wavelet(series):
    c_a, c_d = pywt.dwt(series, 'db5')
    var_a = np.nanvar(c_a)
    var_d = np.nanvar(c_d)
    return [var_a, var_d]


def get_features(series):
    crossings = calculate_crossings(series)
    statistics = calculate_statistics(series)
    var_c_d = var_wavelet(series)
    fd = calc_fractal_dimension(series)
    energy = calc_energy(series)
    return crossings + statistics + var_c_d + fd + energy


if __name__ == "__main__":
    df = pd.read_csv("/home/ashish/Results/Datasets/Shimmer/MP/SegmentedCoordinates/P4_A.csv")
    single_signal = np.array(df[df["sample_id"] == 1]["B8E3_Waist_Accel_LN_X"])
    print(single_signal)
    df.head()

"""
s using the gradient descent algorithm as developed by Madgwick et. al [19] which resulted in
the quaternion W, X, Y and Z signals.

https://github.com/gilestrolab/pyrem/blob/master/src/pyrem/univariate.py
https://github.com/taspinar/siml/blob/master/notebooks/WV4%20-%20Classification%20of%20ECG%20signals%20using%20the%20Discrete%20Wavelet%20Transform.ipynb
https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
"""
