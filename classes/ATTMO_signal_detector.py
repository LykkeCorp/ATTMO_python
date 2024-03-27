import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from DcOS_TrendGenerator import *


class ATTMO_signal_detector:
    __slots__ = ['timeHorizon', 'thresholdsForSignalDetector',
        'currentEventsSignalDetector', 'currentPriceLevelsSignalDetector',
        'previousEventsSignalDetector', 'previousPriceLevelsSignalDetector',
        'numberOfEventsInBlock', #'eventsOscillators',
        'direction', 'dcL', 'osL', 'totalMove',
        'extreme', 'prev_extremes', 'lastDC',
        'textreme', 'tprev_extremes', 'tlastDC',
        'ref', 'nos', 'tmp_mode', 'direction_tmp',
        'dcL_tmp', 'osL_tmp', 'totalMove_tmp',
        'extreme_tmp', 'prev_extremes_tmp',
        'lastDC_tmp', 'textreme_tmp',
        'tprev_extremes_tmp', 'tlastDC_tmp',
        'ref_tmp', 'nos_tmp', 'threshold',
        'signalDetected',
        'totOscillatorBonus', 'ongoingSignalLevel',
        'startingValuesTrendStrength', 'trendForecastLabels',
        'trendStrength', 'trendForecast',
        'alphaParameterExpFunction', 'betaParameterExpFunction',
        'attmoForecastLabels', 'attmoForecast',
        'iterationPredictionStart', 'midpricePredictionStart',
        'iterationPredictionEnd', 'midpricePredictionEnd',
        'predictionDelta', 'target', 'stopLoss', 'distanceToTarget', 'distanceToStopLoss',
        'predictionDurationTicks', 'predictionOutcome', 'ongoingPredictionLevel',
        'nTargetReached', 'nStopLossReached', 'timestampPredictionStart', 'timestampPredictionEnd',
        'timePredictionOn', 'timePredictionOff',
        'overShootDataframe',
        'supportLineDataFrame', 'supportLineEstimationIteration', 'supportLineIntercept', 'supportLineSlope', 'supportLineRSquared',
        'supportLineEstimationPoints', 'supportLineBrakeout', 'supportLineConfirmation',
        'resistenceLineDataFrame', 'resistenceLineEstimationIteration', 'resistenceLineIntercept', 'resistenceLineSlope', 'resistenceLineRSquared',
        'resistenceLineEstimationPoints', 'resistenceLineBrakeout', 'resistenceLineConfirmation']
    def __init__(self, config, t):
        self.timeHorizon = config.timeHorizons[t]
        self.thresholdsForSignalDetector = config.thresholdsForInterpolation[t+5:t+8] #list(np.zeros(3))
        self.currentEventsSignalDetector = list(np.zeros(3))
        self.currentPriceLevelsSignalDetector = list(np.zeros(3))
        self.previousEventsSignalDetector = list(np.zeros(3))
        self.previousPriceLevelsSignalDetector = list(np.zeros(3))
        self.numberOfEventsInBlock = list(np.zeros(3))
        #self.eventsOscillators = list(np.zeros(3))
        self.direction = [[] for _ in range(3)]
        self.dcL = [[] for _ in range(3)]
        self.osL = [[] for _ in range(3)]
        self.totalMove = [[] for _ in range(3)]
        self.extreme = [[] for _ in range(3)]
        self.prev_extremes = [[] for _ in range(3)]
        self.lastDC = [[] for _ in range(3)]
        self.textreme = [[] for _ in range(3)]
        self.tprev_extremes = [[] for _ in range(3)]
        self.tlastDC = [[] for _ in range(3)]
        self.ref = [[] for _ in range(3)]
        self.nos = [[] for _ in range(3)]
        self.tmp_mode = [[] for _ in range(3)]
        self.direction_tmp = [[] for _ in range(3)]
        self.dcL_tmp = [[] for _ in range(3)]
        self.osL_tmp = [[] for _ in range(3)]
        self.totalMove_tmp = [[] for _ in range(3)]
        self.extreme_tmp = [[] for _ in range(3)]
        self.prev_extremes_tmp = [[] for _ in range(3)]
        self.lastDC_tmp = [[] for _ in range(3)]
        self.textreme_tmp = [[] for _ in range(3)]
        self.tprev_extremes_tmp = [[] for _ in range(3)]
        self.tlastDC_tmp = [[] for _ in range(3)]
        self.ref_tmp = [[] for _ in range(3)]
        self.nos_tmp = [[] for _ in range(3)]
        self.threshold = [[] for _ in range(3)]
        self.signalDetected = 0
        self.ongoingPredictionLevel = 0
        self.startingValuesTrendStrength = config.startingValuesTrendStrength
        self.trendForecastLabels = config.trendForecastLabels
        self.trendStrength = 50
        self.trendForecast = "NONE"
        self.alphaParameterExpFunction = 0
        self.betaParameterExpFunction = 0
        self.attmoForecastLabels = config.attmoForecastLabels
        self.attmoForecast = "Foggy"
        self.iterationPredictionStart = 0
        self.midpricePredictionStart = 0
        self.iterationPredictionEnd = 0
        self.midpricePredictionEnd = 0
        self.predictionDelta = 0
        self.target = 0
        self.stopLoss = 0
        self.distanceToTarget = 0
        self.distanceToStopLoss = 0
        self.predictionDurationTicks = 0
        self.predictionOutcome = 0
        self.ongoingPredictionLevel = 0
        self.nTargetReached = 0
        self.nStopLossReached = 0
        self.timestampPredictionStart = ""
        self.timestampPredictionEnd = ""
        self.timePredictionOn = 0
        self.timePredictionOff = 0
        self.overShootDataframe = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
        self.supportLineDataFrame = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
        self.supportLineEstimationIteration = 0
        self.supportLineIntercept = 0
        self.supportLineSlope = 0
        self.supportLineRSquared = 0
        self.supportLineEstimationPoints = 0
        self.supportLineBrakeout = 0
        self.supportLineConfirmation = 0
        self.resistenceLineDataFrame = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
        self.resistenceLineEstimationIteration = 0
        self.resistenceLineIntercept = 0
        self.resistenceLineSlope = 0
        self.resistenceLineRSquared = 0
        self.resistenceLineEstimationPoints = 0
        self.resistenceLineBrakeout = 0
        self.resistenceLineConfirmation = 0

        self.startingValuesTrendStrength = [10, 25, 40, 50, 60, 75, 90]
        self.trendForecastLabels = ['bearish_very_extended', 'bearish_extended', 'bearish_very', 'bearish', 'bearish_neutral',
                                  'bullish_neutral', 'bullish', 'bullish_very', 'bullish_extended', 'bullish_very_extended']
        self.attmoForecastLabels = ['Stormy', 'Rainy', 'Cloudy', 'Foggy', 'Mostly sunny', 'Sunny', 'Tropical']

    def update(self, config, tickReader, dcosSignalDetector, closePrice, iterationBlock, block, columnNamesSignalDetector, foldernameSignalDetector, columnNamesPredictionGeneration, foldernamePredictionsGeneration, columnNamesPredictionOutcome, foldernamePredictionsOutcome):
        midprice = closePrice.getMid()
        tempEvents = list(np.zeros(3))
        self.signalDetected = 0
        for j in range(3):
            tempEvents[j] = dcosSignalDetector[j].run(closePrice)
            if abs(tempEvents[j]) > 0:
                self.previousEventsSignalDetector[j] = self.currentEventsSignalDetector[j]
                self.previousPriceLevelsSignalDetector[j] = self.currentPriceLevelsSignalDetector[j]
                self.currentEventsSignalDetector[j] = tempEvents[j]
                self.currentPriceLevelsSignalDetector[j] = midprice
                self.numberOfEventsInBlock[j] += 1
                #print(f"j = {j}, self.currentEventsSignalDetector[j] = {self.currentEventsSignalDetector[j]}, self.previousEventsSignalDetector[j] = {self.previousEventsSignalDetector[j]}")
            self.direction[j] = -dcosSignalDetector[j].mode
            self.dcL[j] = dcosSignalDetector[j].dcL
            self.osL[j] = dcosSignalDetector[j].osL
            self.totalMove[j] = dcosSignalDetector[j].totalMove
            self.extreme[j] = dcosSignalDetector[j].extreme.level
            self.prev_extremes[j] = dcosSignalDetector[j].prevExtreme.level
            self.lastDC[j] = dcosSignalDetector[j].DC.level
            self.textreme[j] = dcosSignalDetector[j].extreme.time
            self.tprev_extremes[j] = dcosSignalDetector[j].prevExtreme.time
            self.tlastDC[j] = dcosSignalDetector[j].DC.time
            self.ref[j] = dcosSignalDetector[j].reference.level
            self.nos[j] = dcosSignalDetector[j].nOS
            self.threshold[j] = dcosSignalDetector[j].threshold
        self.totOscillatorBonus = np.ceil((self.currentEventsSignalDetector[0] + (self.currentEventsSignalDetector[1]*2) + (self.currentEventsSignalDetector[2]*3)) / 3)

        if abs(self.ongoingPredictionLevel) > 0:
            self = self.checkOngoingPrediction(config, tickReader, iterationBlock, block, columnNamesPredictionOutcome, foldernamePredictionsOutcome)

        if (abs(self.currentEventsSignalDetector[0])>0) & (self.ongoingPredictionLevel==0):
            self = self.generateCrossingBasedPredictions()

        if abs(self.currentEventsSignalDetector[2]) == 2:
            self.overShootDataframe.loc[len(self.overShootDataframe)] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, self.currentEventsSignalDetector[2]]
            if len(self.overShootDataframe) == 11:
                self.overShootDataframe = self.overShootDataframe.drop([0])
                self.overShootDataframe.reset_index(inplace=True)
                if 'index' in self.overShootDataframe.columns:
                    self.overShootDataframe.drop('index', inplace=True, axis=1)
            #print(f"overShootDataframe:")
            #print(f"{self.overShootDataframe}")
            self = self.evaluateFitOfNewDataToTrendLine()
            self = self.generateTrendLineBasedPrediction(config)
            self = self.detectDerivativeTrendLineSignal(tickReader)

        self = self.computeTrendStrengthAndForecast()
        self = self.generateTargetAndStopLoss(config, tickReader, iterationBlock, block, columnNamesPredictionGenerated, foldernamePredictionsGenerated)
        if config.saveSignalDetectionData:
            self.saveSignalDetection(tickReader, dcosSignalDetector, iterationBlock, block, columnNamesSignalDetector, foldernameSignalDetector)
        return self


    def checkOngoingPrediction(self, config, tickReader, iterationBlock, block, columnNamesPredictionOutcome, foldernamePredictionsOutcome):
        if self.ongoingPredictionLevel > 0:
            if tickReader.midprice>self.target:
                self.nTargetReached += 1
                self.predictionOutcome = 1
            elif tickReader.midprice<self.stopLoss:
                self.nStopLossReached += 1
                self.predictionOutcome = -1
        elif self.ongoingPredictionLevel < 0:
            if tickReader.midprice<self.target:
                self.nTargetReached += 1
                self.predictionOutcome = 1
            elif tickReader.midprice>self.stopLoss:
                self.nStopLossReached += 1
                self.predictionOutcome = -1
        if abs(self.predictionOutcome) > 0:
            self.iterationPredictionEnd = tickReader.iteration
            self.midpricePredictionEnd = tickReader.midprice
            self.timestampPredictionEnd = tickReader.timestamp
            self.predictionDurationTicks = self.iterationPredictionEnd - self.iterationPredictionStart
            if config.verbose:
                print("")
                print("---------------------------------------------------------")
                print(f"Time timeHorizon: {self.timeHorizon}, {tickReader.timestamp}: Prediction terminated.")
                if self.predictionOutcome == 1:
                    print(f"Target = {np.round(self.target,3)}, stop-loss = {np.round(self.stopLoss,3)}, midprice = {np.round(tickReader.midprice,3)}: TARGET REACHED!")
                elif self.predictionOutcome == -1:
                    print(f"Target = {np.round(self.target,3)}, stop-loss = {np.round(self.stopLoss,3)}, midprice = {np.round(tickReader.midprice,3)}: STOP-LOSS REACHED!")
                print(f"Duration = {self.predictionDurationTicks} s.")
                print(f"Number of targets reached = {self.nTargetReached}; Number of stop-losses reached = {self.nStopLossReached}.")
            if config.savePredictionData:
                self.savePredictionOutcome(tickReader, iterationBlock, block, columnNamesPredictionOutcome, foldernamePredictionsOutcome)
            if abs(self.ongoingPredictionLevel) > 1:
                self.supportLineDataFrame = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
                self.supportLineRSquared = 0
                self.supportLineEstimationPoints = 0
                self.supportLineIntercept = 0
                self.supportLineSlope = 0
                self.supportLineBrakeout = 0
                self.supportLineConfirmation = 0
                self.resistenceLineDataFrame = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
                self.resistenceLineRSquared = 0
                self.resistenceLineEstimationPoints = 0
                self.resistenceLineIntercept = 0
                self.resistenceLineSlope = 0
                self.resistenceLineBrakeout = 0
                self.resistenceLineConfirmation = 0
            self.ongoingPredictionLevel = 0
            self.predictionOutcome = 0
            self.target = 0
            self.stopLoss = 0
            self.distanceToTarget = 0
            self.distanceToStopLoss = 0
            self.iterationPredictionStart = 0
            self.midpricePredictionStart = 0
            self.timestampPredictionStart = ''
            self.iterationPredictionEnd = 0
            self.midpricePredictionEnd = 0
            self.timestampPredictionEnd = ''
            self.predictionDurationTicks = 0
        return self

    def generateCrossingBasedPredictions(self):
        if (self.previousPriceLevelsSignalDetector[0]<self.currentPriceLevelsSignalDetector[1]<self.currentPriceLevelsSignalDetector[0]) & (self.previousPriceLevelsSignalDetector[0]<self.currentPriceLevelsSignalDetector[2]<self.currentPriceLevelsSignalDetector[0]) & (self.ongoingPredictionLevel==0):
            self.signalDetected = 1
            #if self.ongoingPredictionLevel == 0:
            #    self.ongoingPredictionLevel = 1
        elif (self.previousPriceLevelsSignalDetector[0]>self.currentPriceLevelsSignalDetector[1]>self.currentPriceLevelsSignalDetector[0]) & (self.previousPriceLevelsSignalDetector[0]>self.currentPriceLevelsSignalDetector[2]>self.currentPriceLevelsSignalDetector[0]) & (self.ongoingPredictionLevel==0):
            self.signalDetected = -1
            #if self.ongoingPredictionLevel == 0:
            #    self.ongoingPredictionLevel = -1
        return self

    def evaluateFitOfNewDataToTrendLine(self):
        if (self.supportLineRSquared>0):
            self.evaluateFitOfNewDataToSpportLine()
        if (self.resistenceLineRSquared>0):
            self.evaluateFitOfNewDataToResistenceLine()
        return self

    def evaluateFitOfNewDataToSpportLine(self):
        #print("self.supportLineDataFrame:")
        #print(self.supportLineDataFrame)
        tmp_df = self.supportLineDataFrame.copy()
        tmp_df.reset_index(inplace=True)
        if 'index' in self.overShootDataframe.columns:
            tmp_df.drop('index', inplace=True, axis=1)
        tmp_df.loc[len(tmp_df)] = self.overShootDataframe.iloc[len(self.overShootDataframe)-1]
        subset_df = tmp_df.copy()
        subset_df.set_index('iteration', inplace=True)
        if 'index' in subset_df.columns:
            subset_df.drop('index', inplace=True, axis=1)
        subset = subset_df.midprice
        X = np.arange(len(subset)).reshape(-1, 1)
        y = subset.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r_squared = compute_r_squared(y, y_pred)
        print(f"Fitted new observation to support line. New R-squared: {np.round(r_squared,3)}, new intercept = {np.round(model.intercept_[0],3)}, new slope = {np.round(model.coef_[0],3)}, data:)")
        print(subset_df)
        if (self.supportLineRSquared - r_squared) > 0.1:
            if tmp_df.midprice.iloc[len(tmp_df)-1] < tmp_df.midprice.iloc[len(tmp_df)-2]:
                self.supportLineBrakeout = 1
                print(f"supportLine Brakeout confirmed!")
            elif tmp_df.midprice.iloc[len(tmp_df)-1] > tmp_df.midprice.iloc[len(tmp_df)-2]:
                self.supportLineConfirmation = 1
                print(f"supportLine Confirmation confirmed!")
        return self

    def evaluateFitOfNewDataToResistenceLine(self):
        #print("self.resistenceLineDataFrame:")
        #print(self.resistenceLineDataFrame)
        tmp_df = self.resistenceLineDataFrame.copy()
        tmp_df.reset_index(inplace=True)
        if 'index' in tmp_df.columns:
            tmp_df.drop('index', inplace=True, axis=1)
        tmp_df.loc[len(tmp_df)] = self.overShootDataframe.iloc[len(self.overShootDataframe)-1]
        subset_df = tmp_df.copy()
        subset_df.set_index('iteration', inplace=True)
        if 'index' in subset_df.columns:
            subset_df.drop('index', inplace=True, axis=1)
        subset = subset_df.midprice
        X = np.arange(len(subset)).reshape(-1, 1)
        y = subset.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r_squared = compute_r_squared(y, y_pred)
        print(f"Fitted new observation to resistence line. New R-squared: {np.round(r_squared,3)}, new intercept = {np.round(model.intercept_[0],3)}, new slope = {np.round(model.coef_[0],3)}")
        if (self.resistenceLineRSquared - r_squared) > 0.1:
            if tmp_df.midprice.iloc[len(tmp_df)-1] < tmp_df.midprice.iloc[len(tmp_df)-2]:
                self.resistenceLineConfirmation = 1
                print(f"resistenceLine Brakeout confirmed!")
            elif tmp_df.midprice.iloc[len(tmp_df)-1] > tmp_df.midprice.iloc[len(tmp_df)-2]:
                self.resistenceLineBrakeout = 1
                print(f"resistenceLine Confirmation confirmed!")
        return self

    def generateTrendLineBasedPrediction(self, config):
        if self.supportLineBrakeout | self.resistenceLineConfirmation:
            if abs(self.ongoingPredictionLevel) == 1:
                if config.savePredictionData:
                    self.savePredictionOutcome(tickReader, iterationBlock, block, columnNamesPredictionOutcome, foldernamePredictionsOutcome)
            self.signalDetected = -2
            #self.ongoingPredictionLevel = -2
        if self.supportLineConfirmation | self.resistenceLineBrakeout:
            if abs(self.ongoingPredictionLevel) == 1:
                if config.savePredictionData:
                    self.savePredictionOutcome(tickReader, iterationBlock, block, columnNamesPredictionOutcome, foldernamePredictionsOutcome)
            self.signalDetected = 2
            #self.ongoingPredictionLevel = 2
        if self.supportLineBrakeout & self.resistenceLineConfirmation:
            self.signalDetected = -3
            #self.ongoingPredictionLevel = -3
        if self.supportLineConfirmation & self.resistenceLineBrakeout:
            self.signalDetected = 3
            #self.ongoingPredictionLevel = 3
        #print("")
        #print(f"self.ongoingPredictionLevel = {self.ongoingPredictionLevel}")
        #print("")
        return self

    def detectDerivativeTrendLineSignal(self, tickReader):
        df = self.overShootDataframe.copy()
        if df.direction.iloc[len(df)-1] < 0:
            df = df[df.direction<0]
        elif df.direction.iloc[len(df)-1] > 0:
            df = df[df.direction>0]
        if len(df) > 2:
            for k in range(3,len(df)):
                subset_df = df.iloc[len(df)-k:len(df)]
                #print(f"subset_df = {subset_df}")
                subset_df.set_index('iteration', inplace=True)
                if 'index' in subset_df.columns:
                    subset_df.drop('index', inplace=True, axis=1)
                #print(f"subset_df = {subset_df}")
                subset = subset_df.midprice
                X = np.arange(len(subset)).reshape(-1, 1)
                y = subset.values.reshape(-1, 1)
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r_squared = compute_r_squared(y, y_pred)
                if (df.direction.iloc[len(df)-1]<0) & (r_squared>self.supportLineRSquared) & (r_squared>.9) & (self.supportLineBrakeout==0) & (self.supportLineConfirmation==0):
                    self.supportLineDataFrame = subset_df
                    self.supportLineIntercept = model.intercept_[0]
                    self.supportLineSlope = model.coef_[0][0]
                    self.supportLineRSquared = r_squared
                    self.supportLineEstimationPoints = k
                    self.supportLineEstimationIteration = tickReader.iteration
                    print(f"Support line (positive OS derivative) updated!")
                    print(f"Number of estimation points: {self.supportLineEstimationPoints}.")
                    print(f"Regression R^2: {np.round(self.supportLineRSquared,3)}; inercept: {np.round(self.supportLineIntercept,3)}; slope: {np.round(self.supportLineSlope,3)}")
                    #print(f"Data:")
                    #print(f"{self.supportLineDataFrame}")
                elif (df.direction.iloc[len(df)-1]>0) & (r_squared>self.resistenceLineRSquared) & (r_squared>.9) & (self.resistenceLineBrakeout==0) & (self.resistenceLineConfirmation==0):
                    self.resistenceLineDataFrame = subset_df
                    self.resistenceLineIntercept = model.intercept_[0]
                    self.resistenceLineSlope = model.coef_[0][0]
                    self.resistenceLineRSquared = r_squared
                    self.resistenceLineEstimationPoints = k
                    self.resistenceLineEstimationIteration = tickReader.iteration
                    print(f"Resistence line (negative OS derivative) updated!")
                    print(f"Number of estimation points: {self.resistenceLineEstimationPoints}.")
                    print(f"Regression R^2: {np.round(self.resistenceLineRSquared,3)}; inercept: {np.round(self.resistenceLineIntercept,3)}; slope: {np.round(self.resistenceLineSlope,3)}")
                    #print(f"Data:")
                    #print(f"{self.resistenceLineDataFrame}")
        return self

    def computeTrendStrengthAndForecast(self):
        self.trendStrength = self.startingValuesTrendStrength[self.ongoingPredictionLevel + 3]
        self.trendStrength += self.totOscillatorBonus
        if self.trendStrength > 100:
            self.trendStrength = 100
        elif self.trendStrength < 1:
            self.trendStrength = 1
        self.trendForecast = self.trendForecastLabels[int(np.floor(self.trendStrength/10))]
        self.attmoForecast = self.attmoForecastLabels[self.ongoingPredictionLevel + 3]
        #print(f"self.trendStrength = {self.trendStrength}")
        #print(f"self.trendForecast = {self.trendForecast}")
        #print(f"self.attmoForecast = {self.attmoForecast}")
        return self

    def generateTargetAndStopLoss(self, config, tickReader, iterationBlock, block, columnNamesPredictionGenerated, foldernamePredictionsGenerated):
        if abs(self.ongoingPredictionLevel) > 0:
            self.timePredictionOn += 1
            self.predictionDurationTicks = tickReader.iteration - self.iterationPredictionStart
            self.distanceToTarget = abs(self.target-tickReader.midprice)
            self.distanceToStopLoss = abs(self.stopLoss-tickReader.midprice)
        else:
            self.timePredictionOff += 1
        if abs(self.signalDetected) > abs(self.ongoingPredictionLevel):
            if abs(self.ongoingPredictionLevel) > 0:
                if config.savePredictionData:
                    self.savePredictionOutcome(tickReader, iterationBlock, block, columnNamesPredictionOutcome, foldernamePredictionsOutcome)
            self.ongoingPredictionLevel = self.signalDetected
            self.iterationPredictionStart = tickReader.iteration
            self.midpricePredictionStart = tickReader.midprice
            self.timestampPredictionStart = tickReader.timestamp
            self.predictionDelta = self.threshold[1] * config.predictionFactor
            if self.signalDetected < 0:
                predictionDirection = -1
                #self.direction = -1
            elif self.signalDetected > 0:
                predictionDirection = 1
                #self.direction = 1
            self.target = tickReader.midprice+(tickReader.midprice*self.thresholdsForSignalDetector[1]*config.predictionFactor*predictionDirection)
            self.stopLoss = tickReader.midprice-(tickReader.midprice*self.thresholdsForSignalDetector[1]*config.predictionFactor*predictionDirection)
            if config.verbose:
                print("")
                print(f"---------------------------------------------------------")
                print(f"Time timeHorizon: {self.timeHorizon}, {tickReader.timestamp}: prediction generated!")
                print(f"New signal = {self.signalDetected}. ATTMO forecast = {self.attmoForecast}. Trend strength = {self.trendStrength}. Trend forecast = {self.trendForecast}.")
                print(f"Entry price = {tickReader.midprice}, Target = {np.round(self.target,3)}, StopLoss = {np.round(self.stopLoss,3)} {tickReader.symbol_2}")
            if config.savePredictionData:
                self.savePredictionGenerated(tickReader, iterationBlock, block, columnNamesPredictionGenerated, foldernamePredictionsGenerated)
                print("pred gen saved")
        return self

    def savePredictionOutcome(self, tickReader, iterationBlock, block, columnNamesPredictionOutcome, foldernamePredictionsOutcome):
        dfPredOut = pd.DataFrame(columns=columnNamesPredictionOutcome)
        dfPredOut.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice,  iterationBlock, block,
            self.iterationPredictionStart, self.midpricePredictionStart, self.timestampPredictionStart,
            self.attmoForecast, self.target, self.stopLoss,
            self.predictionDurationTicks, self.predictionOutcome, self.nTargetReached, self.nStopLossReached]
        dfPredOut.to_parquet(f"{foldernamePredictionsOutcome}{tickReader.timestamp}_predictionOutcome.parquet")

    def saveSignalDetection(self, tickReader, dcosSignalDetector, iterationBlock, block, columnNamesSignalDetector, foldernameSignalDetector):
        signalDetector_core = [(self.thresholdsForSignalDetector[i], self.currentEventsSignalDetector[i], self.numberOfEventsInBlock[i]) for i in range(len(dcosSignalDetector))]
        df_signalDetector_core = [item for t in signalDetector_core for item in t]
        df_signalDetector = pd.DataFrame(columns=columnNamesSignalDetector)
        df_signalDetector.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, iterationBlock, block] + df_signalDetector_core + [self.signalDetected, self.ongoingPredictionLevel, self.trendStrength, self.trendForecast, self.attmoForecast, self.supportLineEstimationIteration, self.supportLineIntercept, self.supportLineSlope, self.supportLineRSquared, self.supportLineEstimationPoints, self.supportLineBrakeout, self.supportLineConfirmation,  self.resistenceLineEstimationIteration, self.resistenceLineIntercept, self.resistenceLineSlope, self.resistenceLineRSquared, self.resistenceLineEstimationPoints, self.resistenceLineBrakeout, self.resistenceLineConfirmation]
        df_signalDetector.to_parquet(f"{foldernameSignalDetector}{tickReader.timestamp}_signalDetector.parquet")

    def savePredictionGenerated(self, tickReader, iterationBlock, block, columnNamesPredictionGenerated, foldernamePredictionsGenerated):
        dfPredGen = pd.DataFrame(columns=columnNamesPredictionGenerated)
        dfPredGen.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice,  iterationBlock, block,
            self.predictionDelta*100, self.ongoingPredictionLevel, self.attmoForecast, self.target, self.stopLoss,
            self.predictionDurationTicks, self.predictionOutcome, self.nTargetReached, self.nStopLossReached]
        dfPredGen.to_parquet(f"{foldernamePredictionsGenerated}{tickReader.timestamp}_predictionGenerated.parquet")

    def reset(self, dcosSignalDetector, closePrice, interpolated_deltas):
        self.numberOfEventsInBlock = list(np.zeros(3))
        for j in range(3):
            self.currentEventsSignalDetector[j] = dcosSignalDetector[j].run(closePrice)
            self.direction_tmp[j] = -dcosSignalDetector[j].mode
            self.tmp_mode[j] = dcosSignalDetector[j].mode
            self.dcL_tmp[j] = dcosSignalDetector[j].dcL
            self.osL_tmp[j] = dcosSignalDetector[j].osL
            self.totalMove_tmp[j] = dcosSignalDetector[j].totalMove
            self.extreme_tmp[j] = dcosSignalDetector[j].extreme.level
            self.prev_extremes_tmp[j] = dcosSignalDetector[j].prevExtreme.level
            self.lastDC_tmp[j] = dcosSignalDetector[j].DC.level
            self.textreme_tmp[j] = dcosSignalDetector[j].extreme.time
            self.tprev_extremes_tmp[j] = dcosSignalDetector[j].prevExtreme.time
            self.tlastDC_tmp[j] = dcosSignalDetector[j].DC.time
            self.ref_tmp[j] = dcosSignalDetector[j].reference.level
            self.nos_tmp[j] = dcosSignalDetector[j].nOS
            # reset
            self.thresholdsForSignalDetector[j] = dcosSignalDetector[j].threshold =  interpolated_deltas[j]
            dcosSignalDetector[j] = DcOS_TrendGenerator.DcOS(interpolated_deltas[j], self.tmp_mode[j])
            self.currentEventsSignalDetector[j] = dcosSignalDetector[j].run(closePrice)      # !!!
            dcosSignalDetector[j].mode = self.tmp_mode[j]
            dcosSignalDetector[j].dcL = self.dcL_tmp[j]
            dcosSignalDetector[j].osL = self.osL_tmp[j]
            dcosSignalDetector[j].totalMove = self.totalMove_tmp[j]
            dcosSignalDetector[j].extreme.level = self.extreme_tmp[j]
            dcosSignalDetector[j].prevExtreme.level = self.prev_extremes_tmp[j]
            dcosSignalDetector[j].DC.level = self.lastDC_tmp[j]
            dcosSignalDetector[j].extreme.time = self.textreme_tmp[j]
            dcosSignalDetector[j].prevExtreme.time = self.tprev_extremes_tmp[j]
            dcosSignalDetector[j].DC.time = self.tlastDC_tmp[j]
            dcosSignalDetector[j].reference.level = self.ref_tmp[j]
            dcosSignalDetector[j].nOS = self.nos_tmp[j]
        return self, dcosSignalDetector



def compute_r_squared(y_true, y_pred):
    residual = y_true - y_pred
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum(residual**2)
    r_squared = 1 - (ss_res / ss_total)
    return r_squared
