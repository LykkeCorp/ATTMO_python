import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from DcOS_TrendGenerator import *


class attmoInterpolator:
    __slots__ = ['timeHorizonIndex', 'timeHorizon',
        'thresholdsForInterpolation', 'desiredEventFrequencies',
        'blockLength', 'iterationBlock', 'block',
        'nrOfEventsInBlock', 'currentEventsInterpolator', 'interpolatedThresholds',
        'alphaParameterExpFunction', 'betaParameterExpFunction', 'rSquaredExpFunction',
        'windLabels', 'windLevel', 'windLabel',
        'saveData', 'colNamesDf', 'outputDir', 'plotData', 'outputDirImgs',
        'verbose']
    def __init__(self, config, timeHorizonIndex, colNames_interp, foldernameInterpolation, foldernameImagesInterpolation):
        self.timeHorizonIndex = timeHorizonIndex
        self.timeHorizon = config.timeHorizons[timeHorizonIndex]
        self.thresholdsForInterpolation = np.array(config.thresholdsForInterpolation)
        self.desiredEventFrequencies = np.array(config.desiredEventFrequenciesList[timeHorizonIndex])
        self.blockLength = config.blockLengths[timeHorizonIndex]
        self.iterationBlock = 0
        self.block = 0
        self.nrOfEventsInBlock = list(np.zeros(len(self.thresholdsForInterpolation)))
        self.currentEventsInterpolator = list(np.zeros(len(self.thresholdsForInterpolation))) #[[] for _ in range(len(self.thresholdsForInterpolation))]
        self.interpolatedThresholds = self.thresholdsForInterpolation[timeHorizonIndex+9:timeHorizonIndex+12]
        self.alphaParameterExpFunction = 0
        self.betaParameterExpFunction = 0
        self.rSquaredExpFunction = 0
        self.windLabels = ['Gentle breeze', 'Moderate wind', 'Strong gusts of wind']
        self.windLabel = 'None'
        self.windLevel = 0
        self.saveData = config.saveInterpolationData
        self.colNamesDf = colNames_interp
        self.outputDir = foldernameInterpolation
        self.plotData = config.plotData
        self.outputDirImgs = foldernameImagesInterpolation
        self.verbose = config.verbose
    def run(self, DCOS_interpolation, close_price):
        self.iterationBlock += 1
        for i in range(len(self.thresholdsForInterpolation)):
            self.currentEventsInterpolator[i] = DCOS_interpolation[i].run(close_price)
            if abs(self.currentEventsInterpolator[i]) > 0:
                self.nrOfEventsInBlock[i] += 1
        return self
    def interpolate(self, tickReader): # , target, stopLoss
        i = next(index for index, value in enumerate(self.nrOfEventsInBlock[:-1]) if self.nrOfEventsInBlock[index + 1] >= value)
        subset_a = self.nrOfEventsInBlock[:i+1]
        y_num = [i for i in subset_a if i > 0]
        x_data = self.thresholdsForInterpolation[:len(y_num)]
        y_data = [np.round(y_num[i] / self.iterationBlock * 100, 5) for i in range(len(y_num))]         # = eventFrequencies of increasing, non-0 values
        expFunctionParameters, _ = curve_fit(exponential_func, x_data, y_data)
        self.alphaParameterExpFunction, self.betaParameterExpFunction = expFunctionParameters
        y_pred = list(exponential_func(np.array(x_data), self.alphaParameterExpFunction, self.betaParameterExpFunction))
        self.rSquaredExpFunction = r2_score(y_data, y_pred)
        y_values_to_find = np.array(self.desiredEventFrequencies) * 100
        self.interpolatedThresholds = [find_threshold_for_event_frequency(y, self.alphaParameterExpFunction, self.betaParameterExpFunction) for y in y_values_to_find]
        if ((self.alphaParameterExpFunction>100) & (abs(self.betaParameterExpFunction)>10000)) | ((self.alphaParameterExpFunction<100) & (abs(self.betaParameterExpFunction)<10000)) | (abs(self.betaParameterExpFunction)<12000):
            self.windLevel = 0
        if (self.alphaParameterExpFunction<100) & (abs(self.betaParameterExpFunction)>10000):
            self.windLevel = 1
        if ((self.alphaParameterExpFunction>100) & (abs(self.betaParameterExpFunction)<10000)) | (abs(self.betaParameterExpFunction)<8500):
            self.windLevel = 2
        self.windLabel = self.windLabels[self.windLevel]
        if self.verbose:
            print("")
            print("---------------------------------------------------------")
            print(f"Time timeHorizon: {self.timeHorizon}, {tickReader.timestamp}: resetting deltas.")
            print(f"Exponential function parameters: a = {np.round(self.alphaParameterExpFunction,2)}, b = {np.round(self.betaParameterExpFunction,2)}. R^2 = {np.round(self.rSquaredExpFunction,2)}.")
            print(f"Desired event frequencies = {y_values_to_find}%; signalDetector thresholds = {np.round(np.array(self.interpolatedThresholds)*100,3)}% price change.")
            print(f"Wind conditions = {self.windLabel}")
            #if target != 0:
            #    print(f"Midprice = {tickReader.midprice}, target = {np.round(target,2)}, dist. to target = {np.round(abs(tickReader.midprice-target),2)}, stopLoss = {np.round(stopLoss,2)}, dist. to stop-loss = {np.round(abs(tickReader.midprice-stopLoss),2)}")
        if self.saveData:
            df_interp = pd.DataFrame(columns=self.colNamesDf)
            df_interp.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, self.iterationBlock, self.block, self.interpolatedThresholds[0], self.interpolatedThresholds[1], self.interpolatedThresholds[2], self.alphaParameterExpFunction, self.betaParameterExpFunction, self.rSquaredExpFunction, self.windLevel, self.windLabel]
            df_interp.to_parquet(f"{self.outputDir}{tickReader.timestamp}_interpolation.parquet")
        if self.plotData:
            plt.scatter(x_data, y_data, color='k', label='Data')
            plt.plot(x_data, y_pred, 'c-', label='Predicted')
            plt.xlabel('DcOS Threshold')
            plt.ylabel('Events Count')
            plt.title(f"Model: y = {np.round(self.alphaParameterExpFunction,3)} * np.exp({np.round(self.betaParameterExpFunction,3)} * x), R-squared = {np.round(self.rSquaredExpFunction,3)}")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{self.outputDirImgs}interpolation_block_{self.block:05}.pdf")
            plt.show()

        self.iterationBlock = 0
        self.block += 1
        self.nrOfEventsInBlock = list(np.zeros(len(self.thresholdsForInterpolation)))
        return self


#def power_law(x, a, b):
#    return a / np.power(x, b)

#def inverse_power_law(y, a, b):
#    return np.power(a / y, 1 / b)

#def logarithmic_func(x, a, b):
#    return a * np.log(b * x)

def exponential_func(x, a, b):
    return a * np.exp(b * x)

def find_threshold_for_event_frequency(y, a, b):
    return np.log(y / a) / b
