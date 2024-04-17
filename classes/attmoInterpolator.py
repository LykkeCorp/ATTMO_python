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
        'alphaParameter', 'betaParameter', 'rSquared',
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
        self.currentEventsInterpolator = list(np.zeros(len(self.thresholdsForInterpolation)))
        self.interpolatedThresholds = self.thresholdsForInterpolation[timeHorizonIndex+9:timeHorizonIndex+12]
        self.alphaParameter = 0
        self.betaParameter = 0
        self.rSquared = 0
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
    def interpolate(self, tickReader):
        eventFrequency = np.array(self.nrOfEventsInBlock)/self.blockLength
        idx_y_start = next((i for i, num in enumerate(eventFrequency) if num < .1), None)
        i = next(index for index, value in enumerate(self.nrOfEventsInBlock[idx_y_start:-1]) if self.nrOfEventsInBlock[idx_y_start+index+1] >= value)
        if eventFrequency[idx_y_start+i+1] == 0:
            idx_y_end = idx_y_start+i
        else:
            idx_y_end = idx_y_start+i+1
        y_data = eventFrequency[idx_y_start:idx_y_end]
        if len(y_data) > 2:
            x_data = np.array(self.thresholdsForInterpolation[idx_y_start:idx_y_end])
            powerLawParameters, _ = curve_fit(power_law, x_data, y_data)
            self.alphaParameter, self.betaParameter = powerLawParameters
            y_pred = power_law(np.array(x_data), powerLawParameters[0], powerLawParameters[1])
            self.rSquared = r2_score(y_data, y_pred)
            self.interpolatedThresholds = [inverse_power_law(y, powerLawParameters[0], powerLawParameters[1]) for y in np.array(self.desiredEventFrequencies)]
            y_wind = .2 - .2*(self.betaParameter-1)**2 + .15*(1-self.rSquared)*10
            if y_wind < .1:
                self.windLevel = 0
            elif .1 <= y_wind < .2:
                self.windLevel = 1
            else:
                self.windLevel = 2
            self.windLabel = self.windLabels[self.windLevel]
            if self.verbose:
                print("")
                print("---------------------------------------------------------")
                print(f"Time timeHorizon: {self.timeHorizon}, {tickReader.timestamp}: resetting deltas.")
                print(f"Exponential function parameters: a = {np.round(self.alphaParameter,2)}, b = {np.round(self.betaParameter,2)}. R^2 = {np.round(self.rSquared,2)}.")
                print(f"Desired event frequencies = {np.array(self.desiredEventFrequencies)*100}%; signalDetector thresholds = {np.round(np.array(self.interpolatedThresholds)*100,3)}% price change.")
            if self.plotData:
                plt.scatter(x_data*100, y_data*100, color='k', label='Data')
                plt.plot(x_data*100, y_pred*100, 'c-', label='Predicted')
                plt.scatter(np.array(self.interpolatedThresholds)*100, np.array(self.desiredEventFrequencies)*100, marker="x", color='k', label='Interpolated DcOS thresholds')
                plt.xlabel('DcOS Threshold (% price change)')
                plt.ylabel('Events frequency (% of samples)')
                plt.title(f"Model: y = {np.round(self.alphaParameter,6)} * x^{np.round(self.betaParameter,3)}, R-squared = {np.round(self.rSquared,3)}")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{self.outputDirImgs}interpolation_block_{self.block:05}.pdf")
                plt.show()
        else:
            if self.verbose:
                print("")
                print("---------------------------------------------------------")
                print(f"Time timeHorizon: {self.timeHorizon}, {tickReader.timestamp}: could not reset deltas. Too few data points to interpolate.")
                print(f"SignalDetector thresholds = {np.round(np.array(self.interpolatedThresholds)*100,3)}% price change.")
        print(f"Wind conditions = {self.windLabel}")
        if self.saveData:
            df_interp = pd.DataFrame(columns=self.colNamesDf)
            df_interp.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, self.iterationBlock, self.block, self.interpolatedThresholds[0], self.interpolatedThresholds[1], self.interpolatedThresholds[2], self.alphaParameter, self.betaParameter, self.rSquared, self.windLevel, self.windLabel]
            df_interp.to_parquet(f"{self.outputDir}{tickReader.timestamp}_interpolation.parquet")
        self.iterationBlock = 0
        self.block += 1
        self.nrOfEventsInBlock = list(np.zeros(len(self.thresholdsForInterpolation)))
        return self


def power_law(x, a, b):
    return a / np.power(x, b)

def inverse_power_law(y, a, b):
    return np.power(a / y, 1 / b)

#def logarithmic_func(x, a, b):
#    return a * np.log(b * x)

def exponential_func(x, a, b):
    return a * np.exp(b * x)

def find_threshold_for_event_frequency(y, a, b):
    return np.log(y / a) / b
