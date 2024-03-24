import numpy as np
import pandas as pd
from DcOS_TrendGenerator import *
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


class ATTMO_interpolator:
    __slots__ = ['timeHorizon', 'thresholdsForInterpolation', 'desiredEventFrequencies',
        'blockLength', 'iterationBlock', 'block',
        'nrOfEventsInBlock', 'currentEventsInterpolator', 'interpolatedThresholds',
        'alphaParameterExpFunction', 'betaParameterExpFunction', 'rSquaredExpFunction',
        'windLabels', 'windLevel', 'windLabel']
    def __init__(self, config, t):
        self.timeHorizon = config.timeHorizons[t]
        self.thresholdsForInterpolation = np.array(config.thresholdsForInterpolation)
        self.desiredEventFrequencies = np.array(config.desiredEventFrequenciesList[t])
        self.blockLength = config.blockLengths[t]
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
        self.alphaParameterExpFunction = 0
        self.betaParameterExpFunction = 0
        self.rSquaredExpFunction = 0
        self.windLabels = ['Gentle breeze', 'Moderate wind', 'Strong gusts of wind']
        self.windLabel = 'None'
        self.windLevel = 0
    def run(self, t, DCOS_interpolation, close_price):
        self.iterationBlock += 1
        for i in range(len(self.thresholdsForInterpolation)):
            self.currentEventsInterpolator[i] = DCOS_interpolation[i].run(close_price)
            if abs(self.currentEventsInterpolator[i]) > 0:
                self.nrOfEventsInBlock[i] += 1
        return self
    def interpolate(self, config, t, ticker, target, stopLoss, colNames_interp, foldername_interpolation):
        i = next(index for index, value in enumerate(self.nrOfEventsInBlock[:-1]) if self.nrOfEventsInBlock[index + 1] >= value)
        subset_a = self.nrOfEventsInBlock[:i+1]
        y_num = [i for i in subset_a if i > 0]
        x_data = config.thresholdsForInterpolation[:len(y_num)]
        y_data = [np.round(y_num[i] / self.iterationBlock * 100, 5) for i in range(len(y_num))]         # = eventFrequencies of increasing, non-0 values
        expFunctionParameters, _ = curve_fit(exponential_func, x_data, y_data)
        self.alphaParameterExpFunction, self.betaParameterExpFunction = expFunctionParameters
        y_pred = list(exponential_func(np.array(x_data), self.alphaParameterExpFunction, self.betaParameterExpFunction))
        self.rSquaredExpFunction = r2_score(y_data, y_pred)
        y_values_to_find = np.array(self.desiredEventFrequencies) * 100
        self.interpolatedThresholds = [find_threshold_for_event_frequency(y, self.alphaParameterExpFunction, self.betaParameterExpFunction) for y in y_values_to_find]
        if ((self.alphaParameterExpFunction>100) & (abs(self.betaParameterExpFunction)<10000)) | (abs(self.betaParameterExpFunction)<8500):
            self.windLevel = 2
        elif (self.alphaParameterExpFunction<100) & (abs(self.betaParameterExpFunction)>10000):
            self.windLevel = 1
        elif ((self.alphaParameterExpFunction>100) & (abs(self.betaParameterExpFunction)>10000)) | ((self.alphaParameterExpFunction<100) & (abs(self.betaParameterExpFunction)<10000)):
            self.windLevel = 0
        self.windLabel = self.windLabels[self.windLevel]
        if config.verbose:
            print("")
            print("---------------------------------------------------------")
            print(f"Time timeHorizon: {self.timeHorizon}, {ticker.timestamp}: resetting deltas.")
            print(f"Exponential function parameters: a = {np.round(self.alphaParameterExpFunction,2)}, b = {np.round(self.betaParameterExpFunction,2)}. R^2 = {np.round(self.rSquaredExpFunction,2)}.")
            print(f"Desired event frequencies = {y_values_to_find}%; signalDetector thresholds = {np.round(np.array(self.interpolatedThresholds)*100,3)}% price change.")
            print(f"Wind conditions = {self.windLabel}")
            if target != 0:
                print(f"Midprice = {ticker.midprice}, target = {np.round(target,2)}, dist. to target = {np.round(abs(ticker.midprice-target),2)}, stopLoss = {np.round(stopLoss,2)}, dist. to stop-loss = {np.round(abs(ticker.midprice-stopLoss),2)}")
            #for y_val, x_val in zip(y_values_to_find, self.interpolatedThresholds):
            #    print(f"For a desired event frequency of {y_val}%, a threshold of {np.round(x_val*100,3)}% price change.")
        if config.saveInterpolationData:
            df_interp = pd.DataFrame(columns=colNames_interp)
            df_interp.loc[0] = [ticker.iteration, ticker.timestamp, ticker.midprice, self.iterationBlock, self.block, self.interpolatedThresholds[0], self.interpolatedThresholds[1], self.interpolatedThresholds[2], self.alphaParameterExpFunction, self.betaParameterExpFunction, self.rSquaredExpFunction, self.windLevel, self.windLabel]
            df_interp.to_parquet(f"{foldername_interpolation}{ticker.timestamp}_interpolation.parquet")
        self.iterationBlock = 0
        self.block += 1
        self.nrOfEventsInBlock = list(np.zeros(len(self.thresholdsForInterpolation)))
        return self


def power_law(x, a, b):
    return a / np.power(x, b)

def inverse_power_law(y, a, b):
    return np.power(a / y, 1 / b)

def logarithmic_func(x, a, b):
    return a * np.log(b * x)

def exponential_func(x, a, b):
    return a * np.exp(b * x)

def find_threshold_for_event_frequency(y, a, b):
    return np.log(y / a) / b
