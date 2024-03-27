import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from DcOS_TrendGenerator import *


class ATTMO_signal_detector:
    __slots__ = ['timeHorizon', 'thresholdsForSignalDetector',
        'currentEventsSignalDetector', 'currentPriceLevelsSignalDetector',
        'previousEventsSignalDetector', 'previousPriceLevelsSignalDetector',
        'numberOfEventsInBlock', 'eventsOscillators',
        'direction', 'dcL', 'osL', 'totalMove',
        'extreme', 'prev_extremes', 'lastDC',
        'textreme', 'tprev_extremes', 'tlastDC',
        'ref', 'nos', 'tmp_mode', 'direction_tmp',
        'dcL_tmp', 'osL_tmp', 'totalMove_tmp',
        'extreme_tmp', 'prev_extremes_tmp',
        'lastDC_tmp', 'textreme_tmp',
        'tprev_extremes_tmp', 'tlastDC_tmp',
        'ref_tmp', 'nos_tmp',
        'signalDetected',
        'totOscillatorBonus', 'ongoingSignalLevel',
        'startingValuesTrendStrength', 'trendForecastLabels',
        'trendStrength', 'trendForecast',
        'alphaParameterExpFunction', 'betaParameterExpFunction',
        'attmoForecastLabels', 'attmoForecast',
        'iterationPremiumPredictionStart', 'midpricePremiumPredictionStart',
        'iterationPremiumPredictionEnd', 'midpricePremiumPredictionEnd',
        'premiumPredictionDelta', 'target', 'stopLoss', 'distanceToTarget', 'distanceToStopLoss',
        'premiumPredictionDurationTicks', 'premiumPredictionOutcome', 'premiumPredictionIsOngoing',
        'nTargetReached', 'nStopLossReached', 'timestampPremiumPredictionStart', 'timestampPremiumPredictionEnd',
        'timePremiumPredictionOn', 'timePremiumPredictionOff',
        'overShootDataframe',
        'supportLineDataFrame', 'supportLineEstimationIteration', 'supportLineIntercept', 'supportLineSlope', 'supportLineRSquared',
        'supportLineEstimationPoints', 'supportLineBrakeout', 'supportLineConfirmation',
        'resistenceLineDataFrame', 'resistenceLineEstimationIteration', 'resistenceLineIntercept', 'resistenceLineSlope', 'resistenceLineRSquared',
        'resistenceLineEstimationPoints', 'resistenceLineBrakeout', 'resistenceLineConfirmation']
    def __init__(self, config, t):
        self.timeHorizon = config.timeHorizons[t]
        self.thresholdsForSignalDetector = list(np.zeros(3))
        self.currentEventsSignalDetector = list(np.zeros(3))
        self.currentPriceLevelsSignalDetector = list(np.zeros(3))
        self.previousEventsSignalDetector = list(np.zeros(3))
        self.previousPriceLevelsSignalDetector = list(np.zeros(3))
        self.numberOfEventsInBlock = list(np.zeros(3))
        self.eventsOscillators = list(np.zeros(3))
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
        self.signalDetected = 0
        self.ongoingSignalLevel = 0
        self.startingValuesTrendStrength = config.startingValuesTrendStrength
        self.trendForecastLabels = config.trendForecastLabels
        self.trendStrength = 50
        self.trendForecast = "NONE"
        self.alphaParameterExpFunction = 0
        self.betaParameterExpFunction = 0
        self.attmoForecastLabels = config.attmoForecastLabels
        self.attmoForecast = "Foggy"
        self.iterationPremiumPredictionStart = 0
        self.midpricePremiumPredictionStart = 0
        self.iterationPremiumPredictionEnd = 0
        self.midpricePremiumPredictionEnd = 0
        self.premiumPredictionDelta = 0
        self.target = 0
        self.stopLoss = 0
        self.distanceToTarget = 0
        self.distanceToStopLoss = 0
        self.premiumPredictionDurationTicks = 0
        self.premiumPredictionOutcome = 0
        self.premiumPredictionIsOngoing = 0
        self.nTargetReached = 0
        self.nStopLossReached = 0
        self.timestampPremiumPredictionStart = ""
        self.timestampPremiumPredictionEnd = ""
        self.timePremiumPredictionOn = 0
        self.timePremiumPredictionOff = 0
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
    def update(self, tickReader, dcosSignalDetector, closePrice):
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
                self.eventsOscillators[j] += tempEvents[j]
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
            self.totOscillatorBonus = np.ceil((self.eventsOscillators[0] + (self.eventsOscillators[1]*2) + (self.eventsOscillators[2]*3)) / 3)
        if abs(tempEvents[2]) == 2:
            self.overShootDataframe.loc[len(self.overShootDataframe)] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, tempEvents[2]]
            if len(self.overShootDataframe) == 11:
                self.overShootDataframe = self.overShootDataframe.drop([0])
                self.overShootDataframe.reset_index(inplace=True)
                self.overShootDataframe.drop('index', inplace=True, axis=1)
            print(f"overShootDataframe:")
            print(f"{self.overShootDataframe}")
        return self
    def detectDerivativeTrendLineSignal(self, iteration):
        if len(self.overShootDataframe) == 10:
            df = self.overShootDataframe.copy()
            #df.set_index('iteration', inplace=True)
            #df.drop('index', inplace=True, axis=1)
            if df.direction.iloc[len(df)-1] < 0:
                df = df[df.direction<0]
            elif df.direction.iloc[len(df)-1] > 0:
                df = df[df.direction>0]
            if len(df) > 2:
                for k in range(3,len(df)):
                    subset_df = df.iloc[len(df)-k:len(df)]
                    subset = subset_df.midprice
                    X = np.arange(len(subset)).reshape(-1, 1)
                    y = subset.values.reshape(-1, 1)
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    r_squared = compute_r_squared(y, y_pred)
                    print(f"r-squared = {r_squared}")
                    if (df.direction.iloc[len(df)-1]<0) & (r_squared>self.supportLineRSquared) & (r_squared>.9):
                        print("(r_squared>self.supportLineRSquared) & (r_squared>.9):")
                        self.supportLineDataFrame = subset_df
                        print(f"k = {k}, self.supportLineDataFrame:")
                        print(f"{self.supportLineDataFrame}")
                        self.supportLineIntercept = model.intercept_[0]
                        self.supportLineSlope = model.coef_[0][0]
                        self.supportLineRSquared = r_squared
                        self.supportLineEstimationPoints = k
                        self.supportLineEstimationIteration = iteration
                        print(f'Support line found! Best R-squared positive overshoots: {self.supportLineRSquared} (associated with k={self.supportEstimationPoints}); a = {self.supportLineIntercept}, b = {self.supportLineSlope}')
                    elif (df.direction.iloc[len(df)-1]>0) & (r_squared>self.resistenceLineRSquared) & (r_squared>.9):
                        print("(r_squared>self.resistenceLineRSquared) & (r_squared>.9):")
                        self.resistenceLineDataFrame = subset_df
                        print(f"k = {k}, self.resistenceLineDataFrame:")
                        print(f"{self.resistenceLineDataFrame}")
                        self.resistenceLineIntercept = model.intercept_[0]
                        self.resistenceLineSlope = model.coef_[0][0]
                        self.resistenceLineRSquared = r_squared
                        self.resistenceLineEstimationPoints = k
                        self.resistenceLineEstimationIteration = iteration
                        print(f'Resistence line found! Best R-squared positive overshoots: {self.resistenceLineRSquared} (associated with k={self.resistenceLineEstimationPoints}); a = {self.resistenceLineIntercept}, b = {self.resistenceLineSlope}')
        if (self.supportLineRSquared>0) & (self.supportLineEstimationIteration<iteration):
            tmp_df = self.supportLineDataFrame.copy()
            tmp_df.reset_index(inplace=True)
            tmp_df.drop('index', inplace=True, axis=1)
            tmp_df.loc[len(tmp_df)] = self.overShootDataframe.iloc[len(self.overShootDataframe)-1]
            subset_df = tmp_df.copy()
            subset_df.set_index('iteration', inplace=True)
            #subset_df.drop('index', inplace=True, axis=1)
            subset = subset_df.midprice
            X = np.arange(len(subset)).reshape(-1, 1)
            y = subset.values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r_squared = compute_r_squared(y, y_pred)
            print(f'Fitted new observation to support line. New R-squared: {r_squared}, new intercept = {model.intercept_[0]}, new slope = {model.coef_[0]}')
            if (self.supportLineRSquared - r_squared) > 0.1:
                if self.supportLineSlope < 0:
                    if tmp_df.midprice.iloc[len(tmp_df)-1] < tmp_df.midprice.iloc[len(tmp_df)-2]:
                        self.supportLineBrakeout = 1
                        self.signalDetected = -3
                    elif tmp_df.midprice.iloc[len(tmp_df)-1] > tmp_df.midprice.iloc[len(tmp_df)-2]:
                        self.supportLineConfirmation = 1
                        self.signalDetected = 3
                elif self.supportLineSlope > 0:
                    if tmp_df.midprice.iloc[len(tmp_df)-1] < tmp_df.midprice.iloc[len(tmp_df)-2]:
                        self.supportLineBrakeout = 1
                        self.signalDetected = -3
                    elif tmp_df.midprice.iloc[len(tmp_df)-1] > tmp_df.midprice.iloc[len(tmp_df)-2]:
                        self.supportLineConfirmation = 1
                        self.signalDetected = 3
                if self.supportLineBrakeout == 1:
                    print(f"supportLine Brakeout confirmed!")
                if self.supportLineConfirmation == 1:
                    print(f"supportLine Confirmation confirmed!")
                self.supportLineDataFrame = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
                self.supportLineRSquared = 0
                self.supportLineEstimationPoints = 0
                self.supportLineIntercept = 0
                self.supportLineSlope = 0
        if (self.resistenceLineRSquared>0) & (self.resistenceLineEstimationIteration<iteration):
            tmp_df = self.resistenceLineDataFrame.copy()
            tmp_df.reset_index(inplace=True)
            tmp_df.drop('index', inplace=True, axis=1)
            tmp_df.loc[len(tmp_df)] = self.overShootDataframe.iloc[len(self.overShootDataframe)-1]
            subset_df = tmp_df.copy()
            subset_df.set_index('iteration', inplace=True)
            #subset_df.drop('index', inplace=True, axis=1)
            subset = subset_df.midprice
            X = np.arange(len(subset)).reshape(-1, 1)
            y = subset.values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r_squared = compute_r_squared(y, y_pred)
            print(f'Fitted new observation to resistence line. New R-squared: {r_squared}, new intercept = {model.intercept_[0]}, new slope = {model.coef_[0]}')
            if (self.resistenceLineRSquared - r_squared) > 0.1:
                if self.resistenceLineSlope < 0:
                    if tmp_df.midprice.iloc[len(tmp_df)-1] < tmp_df.midprice.iloc[len(tmp_df)-2]:
                        self.resistenceLineConfirmation = 1
                        self.signalDetected = -3
                    elif tmp_df.midprice.iloc[len(tmp_df)-1] > tmp_df.midprice.iloc[len(tmp_df)-2]:
                        self.resistenceLineBrakeout = 1
                        self.signalDetected = 3
                elif self.best_slope > 0:
                    if tmp_df.midprice.iloc[len(tmp_df)-1] < tmp_df.midprice.iloc[len(tmp_df)-2]:
                        self.resistenceLineConfirmation = 1
                        self.signalDetected = -3
                    elif tmp_df.midprice.iloc[len(tmp_df)-1] > tmp_df.midprice.iloc[len(tmp_df)-2]:
                        self.resistenceLineBrakeout = 1
                        self.signalDetected = 3
                if self.resistenceLineBrakeout == 1:
                    print(f"resistenceLine Brakeout confirmed!")
                if self.resistenceLineConfirmation == 1:
                    print(f"resistenceLine Confirmation confirmed!")
                self.resistenceLineDataFrame = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
                self.resistenceLineRSquared = 0
                self.resistenceLineEstimationPoints = 0
                self.resistenceLineIntercept = 0
                self.resistenceLineSlope = 0

        if config.saveSignalDetectionData:
            signalDetector_core = [(self.thresholdsForSignalDetector[i], self.currentEventsSignalDetector[i], self.numberOfEventsInBlock[i]) for i in range(len(dcosSignalDetector))]
            df_signalDetector_core = [item for t in signalDetector_core for item in t]
            df_signalDetector = pd.DataFrame(columns=columnNamesSignalDetector)
            df_signalDetector.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, iterationBlock, block] + df_signalDetector_core + [self.signalDetected, self.ongoingSignalLevel, self.trendStrength, self.trendForecast, self.attmoForecast, self.supportLineDataFrame, self.supportLineEstimationIteration, self.supportLineIntercept, self.supportLineSlope, self.supportLineRSquared, self.supportLineEstimationPoints, self.supportLineBrakeout, self.supportLineConfirmation, self.resistenceLineDataFrame, self.resistenceLineEstimationIteration, self.resistenceLineIntercept, self.resistenceLineSlope, self.resistenceLineRSquared, self.resistenceLineEstimationPoints, self.resistenceLineBrakeout, self.resistenceLineConfirmation]
            df_signalDetector.to_parquet(f"{foldernameSignalDetector}{tickReader.timestamp}_signalDetector.parquet")
        return self






    def computeTrendStrengthAndForecast(self, block):
        if block > 0:
            if (self.previousPriceLevelsSignalDetector[0]<self.currentPriceLevelsSignalDetector[1]<self.currentPriceLevelsSignalDetector[0]) & (self.previousPriceLevelsSignalDetector[0]<self.currentPriceLevelsSignalDetector[2]<self.currentPriceLevelsSignalDetector[0]) & (self.premiumPredictionIsOngoing==0):
                if self.ongoingSignalLevel != 3:
                    self.signalDetected = 3
                    self.totOscillatorBonus = np.ceil(self.totOscillatorBonus / 3)
                self.ongoingSignalLevel = 3
                self.trendStrength = self.startingValuesTrendStrength[5]
                self.attmoForecast = self.attmoForecastLabels[5]
            elif (self.previousPriceLevelsSignalDetector[0]>self.currentPriceLevelsSignalDetector[1]>self.currentPriceLevelsSignalDetector[0]) & (self.previousPriceLevelsSignalDetector[0]>self.currentPriceLevelsSignalDetector[2]>self.currentPriceLevelsSignalDetector[0]) & (self.premiumPredictionIsOngoing==0):
                if self.ongoingSignalLevel != -3:
                    self.signalDetected = -3
                    self.totOscillatorBonus = np.ceil(self.totOscillatorBonus / 3)
                self.ongoingSignalLevel = -3
                self.trendStrength = self.startingValuesTrendStrength[0]
                self.attmoForecast = self.attmoForecastLabels[0]
            if abs(self.ongoingSignalLevel) == 1:
                if (self.previousPriceLevelsSignalDetector[0]<self.currentPriceLevelsSignalDetector[2]<self.currentPriceLevelsSignalDetector[0]) & (block>0):
                    if self.ongoingSignalLevel != 2:
                        self.signalDetected = 2
                        self.totOscillatorBonus = np.ceil(self.totOscillatorBonus / 2)
                    self.ongoingSignalLevel = 2
                    self.trendStrength = self.startingValuesTrendStrength[4]
                    self.attmoForecast = self.attmoForecastLabels[4]
                elif (self.previousPriceLevelsSignalDetector[0]>self.currentPriceLevelsSignalDetector[2]>self.currentPriceLevelsSignalDetector[0]) & (block>0):
                    if self.ongoingSignalLevel != -2:
                        self.signalDetected = -2
                        self.totOscillatorBonus = np.ceil(self.totOscillatorBonus / 2)
                    self.ongoingSignalLevel = -2
                    self.trendStrength = self.startingValuesTrendStrength[1]
                    self.attmoForecast = self.attmoForecastLabels[1]
            if abs(self.ongoingSignalLevel) == 0:
                if tempEvents[2] == 1:
                    if self.ongoingSignalLevel != 1:
                        self.signalDetected = 1
                    self.ongoingSignalLevel = 1
                    self.trendStrength = self.startingValuesTrendStrength[3]
                    self.attmoForecast = self.attmoForecastLabels[3]
                elif tempEvents[2] == -1:
                    if self.ongoingSignalLevel != -1:
                        self.signalDetected = -1
                    self.trendStrength = self.startingValuesTrendStrength[2]
                    self.attmoForecast = self.attmoForecastLabels[2]
            if self.ongoingSignalLevel == 0:
                self.trendStrength = 50
            self.trendStrength += self.totOscillatorBonus
        if self.trendStrength > 100:
            self.trendStrength = 100
        elif self.trendStrength < 1:
            self.trendStrength = 1
        if 1 <= self.trendStrength < 11:
            self.trendForecast = self.trendForecastLabels[0]
        elif 11 <= self.trendStrength < 21:
            self.trendForecast = self.trendForecastLabels[1]
        elif 21 <= self.trendStrength < 31:
            self.trendForecast = self.trendForecastLabels[2]
        elif 31 <= self.trendStrength < 41:
            self.trendForecast = self.trendForecastLabels[3]
        elif 41 <= self.trendStrength < 51:
            self.trendForecast = self.trendForecastLabels[4]
        elif 51 <= self.trendStrength < 61:
            self.trendForecast = self.trendForecastLabels[5]
        elif 61 <= self.trendStrength < 71:
            self.trendForecast = self.trendForecastLabels[6]
        elif 71 <= self.trendStrength < 81:
            self.trendForecast = self.trendForecastLabels[7]
        elif 81 <= self.trendStrength < 91:
            self.trendForecast = self.trendForecastLabels[8]
        elif 91 <= self.trendStrength < 101:
            self.trendForecast = self.trendForecastLabels[9]
        return self
    def run(self, config, t, tickReader, dcosSignalDetector, closePrice, windLevel, iterationBlock, block, columnNamesSignalDetector, foldernameSignalDetector, columnNamesPredictionGenerated, foldernamePredictionsGenerated, columnNamesPredictionOutcome, foldernamePredictionsOutcome):
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
                if abs(tempEvents[j]) > 0:
                    self.eventsOscillators[j] += tempEvents[j]
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
            self.totOscillatorBonus = np.ceil((self.eventsOscillators[0] + (self.eventsOscillators[1]*2) + (self.eventsOscillators[2]*3)) / 3)
        if block > 0:
            if (self.previousPriceLevelsSignalDetector[0]<self.currentPriceLevelsSignalDetector[1]<self.currentPriceLevelsSignalDetector[0]) & (self.previousPriceLevelsSignalDetector[0]<self.currentPriceLevelsSignalDetector[2]<self.currentPriceLevelsSignalDetector[0]) & (block>0) & (self.premiumPredictionIsOngoing==0):
                if self.ongoingSignalLevel != 3:
                    self.signalDetected = 3
                    self.totOscillatorBonus = np.ceil(self.totOscillatorBonus / 3)
                self.ongoingSignalLevel = 3
                self.trendStrength = self.startingValuesTrendStrength[5]
                self.attmoForecast = self.attmoForecastLabels[5]
            elif (self.previousPriceLevelsSignalDetector[0]>self.currentPriceLevelsSignalDetector[1]>self.currentPriceLevelsSignalDetector[0]) & (self.previousPriceLevelsSignalDetector[0]>self.currentPriceLevelsSignalDetector[2]>self.currentPriceLevelsSignalDetector[0]) & (block>0) & (self.premiumPredictionIsOngoing==0):
                if self.ongoingSignalLevel != -3:
                    self.signalDetected = -3
                    self.totOscillatorBonus = np.ceil(self.totOscillatorBonus / 3)
                self.ongoingSignalLevel = -3
                self.trendStrength = self.startingValuesTrendStrength[0]
                self.attmoForecast = self.attmoForecastLabels[0]
            if abs(self.ongoingSignalLevel) == 1:
                if (self.previousPriceLevelsSignalDetector[0]<self.currentPriceLevelsSignalDetector[2]<self.currentPriceLevelsSignalDetector[0]) & (block>0):
                    if self.ongoingSignalLevel != 2:
                        self.signalDetected = 2
                        self.totOscillatorBonus = np.ceil(self.totOscillatorBonus / 2)
                    self.ongoingSignalLevel = 2
                    self.trendStrength = self.startingValuesTrendStrength[4]
                    self.attmoForecast = self.attmoForecastLabels[4]
                elif (self.previousPriceLevelsSignalDetector[0]>self.currentPriceLevelsSignalDetector[2]>self.currentPriceLevelsSignalDetector[0]) & (block>0):
                    if self.ongoingSignalLevel != -2:
                        self.signalDetected = -2
                        self.totOscillatorBonus = np.ceil(self.totOscillatorBonus / 2)
                    self.ongoingSignalLevel = -2
                    self.trendStrength = self.startingValuesTrendStrength[1]
                    self.attmoForecast = self.attmoForecastLabels[1]
            if abs(self.ongoingSignalLevel) == 0:
                if tempEvents[2] == 1:
                    if self.ongoingSignalLevel != 1:
                        self.signalDetected = 1
                    self.ongoingSignalLevel = 1
                    self.trendStrength = self.startingValuesTrendStrength[3]
                    self.attmoForecast = self.attmoForecastLabels[3]
                elif tempEvents[2] == -1:
                    if self.ongoingSignalLevel != -1:
                        self.signalDetected = -1
                    self.trendStrength = self.startingValuesTrendStrength[2]
                    self.attmoForecast = self.attmoForecastLabels[2]
            if self.ongoingSignalLevel == 0:
                self.trendStrength = 50
            self.trendStrength += self.totOscillatorBonus
        if self.trendStrength > 100:
            self.trendStrength = 100
        elif self.trendStrength < 1:
            self.trendStrength = 1
        if 1 <= self.trendStrength < 11:
            self.trendForecast = self.trendForecastLabels[0]
        elif 11 <= self.trendStrength < 21:
            self.trendForecast = self.trendForecastLabels[1]
        elif 21 <= self.trendStrength < 31:
            self.trendForecast = self.trendForecastLabels[2]
        elif 31 <= self.trendStrength < 41:
            self.trendForecast = self.trendForecastLabels[3]
        elif 41 <= self.trendStrength < 51:
            self.trendForecast = self.trendForecastLabels[4]
        elif 51 <= self.trendStrength < 61:
            self.trendForecast = self.trendForecastLabels[5]
        elif 61 <= self.trendStrength < 71:
            self.trendForecast = self.trendForecastLabels[6]
        elif 71 <= self.trendStrength < 81:
            self.trendForecast = self.trendForecastLabels[7]
        elif 81 <= self.trendStrength < 91:
            self.trendForecast = self.trendForecastLabels[8]
        elif 91 <= self.trendStrength < 101:
            self.trendForecast = self.trendForecastLabels[9]
        if config.saveSignalDetectionData:
            signalDetector_core = [(self.thresholdsForSignalDetector[i], self.currentEventsSignalDetector[i], self.numberOfEventsInBlock[i]) for i in range(len(dcosSignalDetector))]
            df_signalDetector_core = [item for t in signalDetector_core for item in t]
            df_signalDetector = pd.DataFrame(columns=columnNamesSignalDetector)
            df_signalDetector.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, iterationBlock, block] + df_signalDetector_core + [self.signalDetected, self.ongoingSignalLevel, self.trendStrength, self.trendForecast, self.attmoForecast, self.supportLineDataFrame, self.supportLineEstimationIteration, self.supportLineIntercept, self.supportLineSlope, self.supportLineRSquared, self.supportLineEstimationPoints, self.supportLineBrakeout, self.supportLineConfirmation, self.resistenceLineDataFrame, self.resistenceLineEstimationIteration, self.resistenceLineIntercept, self.resistenceLineSlope, self.resistenceLineRSquared, self.resistenceLineEstimationPoints, self.resistenceLineBrakeout, self.resistenceLineConfirmation]
            df_signalDetector.to_parquet(f"{foldernameSignalDetector}{tickReader.timestamp}_signalDetector.parquet")
        if (config.verbose) & (abs(self.signalDetected)>0) & (block>0):
            print("")
            print(f"---------------------------------------------------------")
            print(f"Time timeHorizon: {self.timeHorizon}, {tickReader.timestamp}: signal detected.")
            print(f"New signal = {self.signalDetected}. ATTMO forecast = {self.attmoForecast}. Trend strength = {self.trendStrength}. Trend forecast = {self.trendForecast}.")
        if abs(self.signalDetected) == 3:
            if self.signalDetected == 3:
                self.premiumPredictionIsOngoing = 1
            elif self.signalDetected == -3:
                self.premiumPredictionIsOngoing = -1
            self.iterationPremiumPredictionStart = tickReader.iteration
            self.midpricePremiumPredictionStart = tickReader.midprice
            self.timestampPremiumPredictionStart = tickReader.timestamp
            self.premiumPredictionDelta = self.thresholdsForSignalDetector[1] * config.predictionFactor
            if block > 0:
                if self.premiumPredictionIsOngoing == 0:
                    self.timePremiumPredictionOff += 1
                else:
                    self.timePremiumPredictionOn += 1
                self.target = tickReader.midprice+(tickReader.midprice*self.premiumPredictionDelta*self.premiumPredictionIsOngoing*(1+windLevel))
                self.stopLoss = tickReader.midprice-(tickReader.midprice*self.premiumPredictionDelta*self.premiumPredictionIsOngoing*(1+windLevel))
                if config.verbose:
                    print("")
                    print(f"---------------------------------------------------------")
                    print(f"Time timeHorizon: {self.timeHorizon}, {tickReader.timestamp}: Premium prediction generated!")
                    print(f"Entry price = {tickReader.midprice}, Target = {np.round(self.target,3)}, StopLoss = {np.round(self.stopLoss,3)} {tickReader.symbol_2}")
                if config.savePredictionData:
                    dfPredGen = pd.DataFrame(columns=columnNamesPredictionGenerated)
                    dfPredGen.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice,  iterationBlock, block,
                        self.premiumPredictionDelta*100, self.premiumPredictionIsOngoing, self.attmoForecast, self.target, self.stopLoss,
                        self.premiumPredictionDurationTicks, self.premiumPredictionOutcome, self.nTargetReached, self.nStopLossReached]
                    dfPredGen.to_parquet(f"{foldernamePredictionsGenerated}{tickReader.timestamp}_predictionGenerated.parquet")
        if self.premiumPredictionIsOngoing == 1:
            if tickReader.midprice>self.target:
                self.nTargetReached += 1
                self.premiumPredictionOutcome = 1
            elif tickReader.midprice<self.stopLoss:
                self.nStopLossReached += 1
                self.premiumPredictionOutcome = -1
        elif self.premiumPredictionIsOngoing == -1:
            if tickReader.midprice<self.target:
                self.nTargetReached += 1
                self.premiumPredictionOutcome = 1
            elif tickReader.midprice>self.stopLoss:
                self.nStopLossReached += 1
                self.premiumPredictionOutcome = -1
        if abs(self.premiumPredictionOutcome) > 0:
            self.iterationPremiumPredictionEnd = tickReader.iteration
            self.midpricePremiumPredictionEnd = tickReader.midprice
            self.timestampPremiumPredictionEnd = tickReader.timestamp
            self.premiumPredictionDurationTicks = self.iterationPremiumPredictionEnd - self.iterationPremiumPredictionStart
            if config.verbose:
                print("")
                print("---------------------------------------------------------")
                print(f"Time timeHorizon: {self.timeHorizon}, {tickReader.timestamp}: Premium prediction terminated.")
                if self.premiumPredictionOutcome == 1:
                    print(f"Target = {np.round(self.target,3)}, stop-loss = {np.round(self.stopLoss,3)}, midprice = {np.round(tickReader.midprice,3)}: TARGET REACHED!")
                elif self.premiumPredictionOutcome == -1:
                    print(f"Target = {np.round(self.target,3)}, stop-loss = {np.round(self.stopLoss,3)}, midprice = {np.round(tickReader.midprice,3)}: STOP-LOSS REACHED!")
                print(f"Duration = {self.premiumPredictionDurationTicks} s.")
                print(f"Number of targets reached = {self.nTargetReached}; Number of stop-losses reached = {self.nStopLossReached}.")
            if config.savePredictionData:
                dfPredOut = pd.DataFrame(columns=columnNamesPredictionOutcome)
                dfPredOut.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice,  iterationBlock, block,
                    self.iterationPremiumPredictionStart, self.midpricePremiumPredictionStart, self.timestampPremiumPredictionStart,
                    self.attmoForecast, self.target, self.stopLoss,
                    self.premiumPredictionDurationTicks, self.premiumPredictionOutcome, self.nTargetReached, self.nStopLossReached]
                dfPredOut.to_parquet(f"{foldernamePredictionsOutcome}{tickReader.timestamp}_predictionOutcome.parquet")
            self.ongoingSignalLevel = 0
            self.attmoForecast = 'Foggy'
            self.premiumPredictionOutcome = 0
            self.premiumPredictionIsOngoing = 0
            self.target = 0
            self.stopLoss = 0
            self.distanceToTarget = 0
            self.distanceToStopLoss = 0
        self.currentEventsSignalDetector = list(np.zeros(3))
        self.signalDetected = 0
        self.eventsOscillators = list(np.zeros(3))
        return self
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
