import numpy as np
import pandas as pd


class ATTMO_prediction_generator:
    __slots__ = ['timeHorizon', 'alphaParameterPowerLaw', 'betaParameterPowerLaw', 'counterTrend',
        'iterationPredictionStart', 'midpricePredictionStart', 'iterationPredictionEnd', 'midpricePredictionEnd',
        'timestampPredictionStart', 'timestampPredictionEnd', 'timeUntilExpiration',
        'predictionDelta', 'direction', 'target', 'stopLoss',
        'predictionDuration', 'predictionOutcome', 'predictionIsOngoing',
        'distanceToTarget', 'distanceToStopLoss', 'nTargetReached', 'nStopLossReached', 'nTargetExpired', 'nStopLossExpired',
        'predictionDurations', 'timePredictionOn', 'timePredictionOff',
        'attmoForecastLabels', 'attmoForecast', 'trendStrength', 'trendForecast']
    def __init__(self, config, t):
        self.timeHorizon = config.timeHorizons[t]
        self.alphaParameterPowerLaw = 0
        self.betaParameterPowerLaw = 0
        self.counterTrend = 0
        self.iterationPredictionStart = 0
        self.midpricePredictionStart = 0
        self.iterationPredictionEnd = 0
        self.midpricePredictionEnd = 0
        self.predictionDelta = 0
        self.direction = 0
        self.target = 0
        self.stopLoss = 0
        self.distanceToTarget = 0
        self.distanceToStopLoss = 0
        self.predictionDuration = 0
        self.predictionOutcome = 0
        self.predictionIsOngoing = 0
        self.nTargetReached = 0
        self.nStopLossReached = 0
        self.nTargetExpired = 0
        self.nStopLossExpired = 0
        self.timestampPredictionStart = ""
        self.timestampPredictionEnd = ""
        self.predictionDurations = []
        self.timePredictionOn = 0
        self.timePredictionOff = 0
        self.attmoForecastLabels = config.attmoForecastLabels
        self.attmoForecast = "NONE"
    def run(self, tickReader, iterationBlock, block, powerLawParameters, signalDetectorMThreshold, currentSignalLevel, config, colNamesPrediction, foldername, foldernamePredictions):
        self.alphaParameterPowerLaw = powerLawParameters[0]
        self.betaParameterPowerLaw = powerLawParameters[1]
        if block > 0:
            if self.predictionIsOngoing == 0:
                self.timePredictionOff += 1
            else:
                self.timePredictionOn += 1
        if self.predictionIsOngoing:
            self.predictionDuration = tickReader.iteration - self.iterationPredictionStart
            self.distanceToTarget = abs(self.target-tickReader.midprice)
            self.distanceToStopLoss = abs(self.stopLoss-tickReader.midprice)
        if (self.predictionIsOngoing==0) & (abs(currentSignalLevel)==3):
            self.predictionIsOngoing = 1
            self.iterationPredictionStart = tickReader.iteration
            self.midpricePredictionStart = tickReader.midprice
            self.timestampPredictionStart = tickReader.timestamp
            self.predictionDelta = signalDetectorMThreshold * config.predictionFactor
            if currentSignalLevel == 3:
                self.direction = -1
            elif currentSignalLevel == -3:
                self.direction = 1
            self.target = tickReader.midprice+(tickReader.midprice*self.predictionDelta*self.direction)
            self.stopLoss = tickReader.midprice-(tickReader.midprice*self.predictionDelta*self.direction)
            if self.predictionIsOngoing:
                if self.alphaParameterPowerLaw < .0005:
                    self.attmoForecast = self.attmoForecastLabels[3]
                elif .0005 <= self.alphaParameterPowerLaw < .001:
                    self.attmoForecast = self.attmoForecastLabels[4]
                elif self.alphaParameterPowerLaw >= .001:
                    self.attmoForecast = self.attmoForecastLabels[5]
            elif self.predictionIsOngoing == -3:
                if self.alphaParameterPowerLaw < .0005:
                    self.attmoForecast = self.attmoForecastLabels[0]
                elif .0005 <= self.alphaParameterPowerLaw < .001:
                    self.attmoForecast = self.attmoForecastLabels[1]
                elif self.alphaParameterPowerLaw >= .001:
                    self.attmoForecast = self.attmoForecastLabels[2]
            if config.verbose:
                print("")
                print(f"---------------------------------------------------------")
                print(f"{tickReader.timestamp}: PREDICTION GENERATOR: Prediction generated!")
                print(f"Midprice = {tickReader.midprice}")
                print(f"ATTMO forecast = {self.attmoForecast}")
                print(f"Target = {np.round(self.target,3)} {tickReader.symbol_2}")
                print(f"StopLoss = {np.round(self.stopLoss,3)} {tickReader.symbol_2}")
        if self.predictionIsOngoing:
            if self.direction == 1:
                if tickReader.midprice>self.target:
                    self.nTargetReached += 1
                    self.predOutcome = 1
                elif tickReader.midprice<self.stopLoss:
                    self.nStopLossReached += 1
                    self.predOutcome = -1
            elif self.direction == -1:
                if tickReader.midprice<self.target:
                    self.nTargetReached += 1
                    self.predOutcome = 1
                elif tickReader.midprice>self.stopLoss:
                    self.nStopLossReached += 1
                    self.predOutcome = -1
        if abs(self.predictionOutcome) > 0:
            self.iterationPredictionEnd = tickReader.iteration
            self.midpricePredictionEnd = tickReader.midprice
            self.timestampPredictionEnd = tickReader.timestamp
            self.predictionDuration = self.iterationPredictionEnd - self.iterationPredictionStart
            if config.verbose:
                print("")
                print("---------------------------------------------------------")
                print(f"{tickReader.timestamp}: PREDICTION GENERATOR: Prediction terminated.")
                if self.predictionOutcome == 1:
                    print(f"Target = {np.round(self.target,3)}, stop-loss = {np.round(self.stopLoss,3)}, midprice = {np.round(tickReader.midprice,3)}: TARGET REACHED!")
                elif self.predictionOutcome == -1:
                    print(f"Target = {np.round(self.target,3)}, stop-loss = {np.round(self.stopLoss,3)}, midprice = {np.round(tickReader.midprice,3)}: STOP-LOSS REACHED!")
                print(f"Number of targets reached = {self.nTargetReached}")
                print(f"Number of stop-losses reached = {self.nStopLossReached}")
        dfPred = pd.DataFrame(columns=colNamesPrediction)
        dfPred.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, iterationBlock, block,
                self.iterationPredictionStart, self.timestampPredictionStart, self.midpricePredictionStart,
                self.iterationPredictionEnd, self.midpricePredictionEnd,  self.timestampPredictionEnd,
                self.predictionDelta*100, self.direction, self.target, self.stopLoss,
                self.attmoForecast, self.predictionDuration,
                self.predictionOutcome, self.nTargetReached, self.nStopLossReached]
        dfPred.to_parquet(f"{foldername}currentPrediction.parquet")
        if abs(self.predictionOutcome) > 0:
            dfPred = pd.DataFrame(columns=colNamesPrediction)
            dfPred.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice,  iterationBlock, block,
                self.iterationPredictionStart, self.timestampPredictionStart, self.midpricePredictionStart,
                self.iterationPredictionEnd, self.midpricePredictionEnd,  self.timestampPredictionEnd,
                self.predictionDelta*100, self.direction, self.target, self.stopLoss,
                self.attmoForecast, self.predictionDuration,
                self.predictionOutcome, self.nTargetReached, self.nStopLossReached]
            dfPred.to_parquet(f"{foldernamePredictions}{tickReader.timestamp}_prediction.parquet")
            self.predictionDurations.append(self.predictionDuration)
            self.iterationPredictionStart = 0
            self.midpricePredictionStart = 0
            self.iterationPredictionEnd = 0
            self.midpricePredictionEnd = 0
            self.predictionDelta = 0
            self.direction = 0
            self.target = 0
            self.stopLoss = 0
            self.attmoForecast = "NONE"
            self.predictionDuration = 0
            self.predictionOutcome = 0
            self.predictionIsOngoing = 0
            self.timestampPredictionStart = ""
            self.timestampPredictionEnd = ""
            self.distanceToTarget = 0
            self.distanceToStopLoss = 0
        return self
