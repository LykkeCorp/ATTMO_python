import numpy as np
import pandas as pd
from DcOS_TrendGenerator import *
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


class ATTMO_interpolator:
    __slots__ = ['timeHorizon', 'thresholdsForInterpolation', 'desiredEventFrequencies',
        'iterationBlock', 'block',
        'nrOfEventsInBlock', 'currentEventsInterpolator', 'interpolatedThresholds',
        'eventFrequencies', 'powerLawParameters', 'powerLawRSquared']
    def __init__(self, config, t):
        self.timeHorizon = config.timeHorizons[t]
        self.thresholdsForInterpolation = np.array(config.thresholdsForInterpolation)
        self.desiredEventFrequencies = np.array(config.desiredEventFrequenciesList[t])
        self.iterationBlock = 0
        self.block = 0
        self.nrOfEventsInBlock = list(np.zeros(len(self.thresholdsForInterpolation)))
        self.currentEventsInterpolator = [[] for _ in range(len(self.thresholdsForInterpolation))]
        if t == 0:
            self.interpolatedThresholds = [self.thresholdsForInterpolation[3], self.thresholdsForInterpolation[4], self.thresholdsForInterpolation[5]]
        elif t == 1:
            self.interpolatedThresholds = [self.thresholdsForInterpolation[4], self.thresholdsForInterpolation[5], self.thresholdsForInterpolation[6]]
        elif t == 2:
            self.interpolatedThresholds = [self.thresholdsForInterpolation[5], self.thresholdsForInterpolation[6], self.thresholdsForInterpolation[7]]
        elif t == 3:
            self.interpolatedThresholds = [self.thresholdsForInterpolation[6], self.thresholdsForInterpolation[7], self.thresholdsForInterpolation[8]]
        elif t == 4:
            self.interpolatedThresholds = [self.thresholdsForInterpolation[7], self.thresholdsForInterpolation[8], self.thresholdsForInterpolation[9]]
        elif t == 5:
            self.interpolatedThresholds = [self.thresholdsForInterpolation[8], self.thresholdsForInterpolation[9], self.thresholdsForInterpolation[10]]
        self.eventFrequencies = 0
        self.powerLawParameters = [0, 0]
        self.powerLawRSquared = 0
    def run(self, t, DCOS_interpolation, close_price):
        self.iterationBlock += 1
        for i in range(len(self.thresholdsForInterpolation)):
            self.currentEventsInterpolator[i] = DCOS_interpolation[i].run(close_price)
            if abs(self.currentEventsInterpolator[i]) > 0:
                self.nrOfEventsInBlock[i] += 1
        return self
    def fit_power_law(self, config, ticker, colNames_interp, foldername_interpolation):
        i = next(index for index, value in enumerate(self.nrOfEventsInBlock[:-1]) if self.nrOfEventsInBlock[index + 1] >= value)
        subset_a = self.nrOfEventsInBlock[:i+1]
        y_num = [i for i in subset_a if i > 0]
        IDXs = list(np.arange(len(y_num)))
        x_data = config.thresholdsForInterpolation[:len(y_num)]
        y_data = [np.round(y_num[i] / self.iterationBlock* 100, 5) for i in range(len(y_num))]
        self.powerLawParameters, _ = curve_fit(power_law, x_data, y_data)
        a_fit, b_fit = self.powerLawParameters
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = power_law(x_fit, a_fit, b_fit)
        y_pred = power_law(x_data, a_fit, b_fit)
        self.powerLawRSquared = r2_score(y_data, y_pred)
        y_diff = y_data - y_pred
        y_diff_abs = abs(y_diff)
        IDX_min = np.where(y_diff_abs == y_diff_abs.min())
        min_diff = y_diff[IDX_min]
        IDX_max = np.where(y_diff_abs == y_diff_abs.max())
        max_diff = y_diff[IDX_max]
        y_values_to_find = np.array(config.desiredEventFrequencies) * 100
        self.interpolatedThresholds = inverse_power_law(y_values_to_find, a_fit, b_fit)
        if config.verbose:
            print(f"{ticker.timestamp}: resetting deltas...")
            for y_val, x_val in zip(y_values_to_find, self.interpolatedThresholds):
                print(f"For a desired event frequency of {y_val}%, a threshold of {np.round(x_val*100,3)}% price change.")
        if config.saveInterpolationData:
            df_interp = pd.DataFrame(columns=colNames_interp)
            df_interp.loc[0] = [ticker.iteration, ticker.timestamp, ticker.midprice, self.iterationBlock, self.block, self.interpolatedThresholds[0], self.interpolatedThresholds[1], self.interpolatedThresholds[2], self.powerLawParameters[0], self.powerLawParameters[1], self.powerLawRSquared]
            df_interp.to_parquet(f"{foldername_interpolation}{ticker.timestamp}_interpolation.parquet")
        self.iterationBlock = 0
        self.block += 1
        self.nrOfEventsInBlock = list(np.zeros(len(self.thresholdsForInterpolation)))
        return self


def power_law(x, a, b):
    return a / np.power(x, b)
def inverse_power_law(y, a, b):
    return np.power(a / y, 1 / b)
