import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from DcOS_TrendGenerator import *


class attmoSignalDetector:
    __slots__ = ['timeHorizon', 'thresholdsForSignalDetector',
        'signalDetected', 'mostRecentPredictionLevel',
        'totOscillatorBonus', 'ongoingSignalLevel',
        'startingValuesTrendStrength', 'trendForecastLabels',
        'trendStrength', 'trendForecast',
        'alphaParameterExpFunction', 'betaParameterExpFunction',
        'attmoForecastLabels', 'attmoForecast',
        'colNamesDf', 'outputDir',
        'plotData', 'outputDirImgs']
    def __init__(self, config, t, columnNamesSignalDetector, foldernameSignalDetector, foldernameImagesSignalDetector):
        self.timeHorizon = config.timeHorizons[t]
        self.thresholdsForSignalDetector = config.thresholdsForInterpolation[t+9:t+12]
        self.signalDetected = 0
        self.mostRecentPredictionLevel = 0
        self.trendStrength = 50
        self.trendForecast = "NONE"
        self.alphaParameterExpFunction = 0
        self.betaParameterExpFunction = 0
        self.attmoForecast = "Foggy"
        self.startingValuesTrendStrength = [10, 25, 40, 50, 60, 75, 90]
        self.trendForecastLabels = ['bearish_very_extended', 'bearish_extended', 'bearish_very', 'bearish', 'bearish_neutral',
                                  'bullish_neutral', 'bullish', 'bullish_very', 'bullish_extended', 'bullish_very_extended']
        self.attmoForecastLabels = ['Stormy', 'Rainy', 'Cloudy', 'Foggy', 'Mostly sunny', 'Sunny', 'Tropical']
        self.colNamesDf = columnNamesSignalDetector
        self.outputDir = foldernameSignalDetector

    def update(self, config, tickReader, dcosTraceGenerator, dcosEventsSignalDetector, predictionGenerator, crossSignal, trendLines, closePrice, iterationBlock, block):
        self.signalDetected = 0

        events = list(np.zeros(3))
        for j in range(3):
            events[j] = dcosTraceGenerator[j].run(closePrice)
        dcosEventsSignalDetector = dcosEventsSignalDetector.update(dcosTraceGenerator, events, closePrice)
        predictionGenerator = predictionGenerator.checkOngoingPredictions(tickReader)

        if (abs(events[0])>0):
            crossSignal = crossSignal.generateSignal(dcosEventsSignalDetector)
            self.signalDetected = crossSignal.signal

        if abs(events[2]) == 2:
            for i in range(2):
                trendLines[i] = trendLines[i].updateAndFitToNewData(tickReader, events[2])
            if ((trendLines[0].signal==2) & (trendLines[1].rSquared>0)) | ((trendLines[1].signal==2) & (trendLines[0].rSquared>0)):
                self.signalDetected = 3
            elif ((trendLines[0].signal==-2) & (trendLines[1].rSquared>0)) | ((trendLines[1].signal==-2) & (trendLines[0].rSquared>0)):
                self.signalDetected = -3
            elif (trendLines[0].signal==2) | (trendLines[1].signal==2):
                self.signalDetected = 2
            elif (trendLines[0].signal==-2) | (trendLines[1].signal==-2):
                self.signalDetected = -2

            if events[2] == -2:
                trendLines[0] = trendLines[0].detectTrendLine(tickReader)
            elif events[2] == 2:
                trendLines[1] = trendLines[1].detectTrendLine(tickReader)

        if block > 0:
            predictionGenerator = predictionGenerator.generatePrediction(self.signalDetected, dcosTraceGenerator[1].threshold, config.predictionFactor, tickReader)

        if abs(self.signalDetected) > 0:
            self.mostRecentPredictionLevel = self.signalDetected
        self.trendStrength = self.startingValuesTrendStrength[self.mostRecentPredictionLevel + 3]
        self.trendStrength += dcosEventsSignalDetector.totOscillatorBonus
        if self.trendStrength > 100:
            self.trendStrength = 100
        elif self.trendStrength < 1:
            self.trendStrength = 1
        self.trendForecast = self.trendForecastLabels[int(np.floor(self.trendStrength/10))]
        self.attmoForecast = self.attmoForecastLabels[self.mostRecentPredictionLevel + 3]

        if config.saveSignalDetectionData:
            self.saveSignalDetection(tickReader, dcosEventsSignalDetector, iterationBlock, block, trendLines)
        return self


    def saveSignalDetection(self, tickReader, dcosEventsSignalDetector, iterationBlock, block, trendLines):
        signalDetector_core = [(self.thresholdsForSignalDetector[i], dcosEventsSignalDetector.currentEvents[i], dcosEventsSignalDetector.numberOfEventsInBlock[i]) for i in range(len(self.thresholdsForSignalDetector))]
        df_signalDetector_core = [item for t in signalDetector_core for item in t]
        df_signalDetector = pd.DataFrame(columns=self.colNamesDf)
        df_signalDetector.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, iterationBlock, block] + df_signalDetector_core + [self.signalDetected, self.mostRecentPredictionLevel, self.trendStrength, self.trendForecast, self.attmoForecast, trendLines[0].estimationIteration, trendLines[0].intercept, trendLines[0].slope, trendLines[0].rSquared, trendLines[0].estimationPoints, trendLines[1].estimationIteration, trendLines[1].intercept, trendLines[1].slope, trendLines[1].rSquared, trendLines[1].estimationPoints]
        df_signalDetector.to_parquet(f"{self.outputDir}{tickReader.timestamp}_signalDetector.parquet")


class intrinsicTimeEventsSignalDetector:
    __slots__ = ['timeHorizon', 'thresholdsForSignalDetector',
        'direction', 'dcL', 'osL', 'totalMove',
        'extreme', 'prev_extremes', 'lastDC',
        'textreme', 'tprev_extremes', 'tlastDC',
        'ref', 'nos', 'tmp_mode', 'direction_tmp',
        'dcL_tmp', 'osL_tmp', 'totalMove_tmp',
        'extreme_tmp', 'prev_extremes_tmp',
        'lastDC_tmp', 'textreme_tmp',
        'tprev_extremes_tmp', 'tlastDC_tmp',
        'ref_tmp', 'nos_tmp', 'threshold_tmp', 'threshold',
        'previousEvents', 'previousPriceLevels',
        'currentEvents', 'currentPriceLevels',
        'numberOfEventsInBlock',
        'totOscillatorBonus']
    def __init__(self, timeHorizon, thresholdsForSignalDetector):
        self.timeHorizon = timeHorizon
        self.thresholdsForSignalDetector = thresholdsForSignalDetector
        self.direction = list(np.zeros(len(self.thresholdsForSignalDetector))) #[[] for _ in range(3)]
        self.dcL = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.osL = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.totalMove = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.extreme = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.prev_extremes = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.lastDC = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.textreme = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.tprev_extremes = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.tlastDC = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.ref = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.nos = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.threshold = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.tmp_mode = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.direction_tmp = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.dcL_tmp = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.osL_tmp = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.totalMove_tmp = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.extreme_tmp = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.prev_extremes_tmp = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.lastDC_tmp = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.textreme_tmp = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.tprev_extremes_tmp = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.tlastDC_tmp = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.ref_tmp = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.nos_tmp = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.threshold_tmp = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.previousEvents = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.previousPriceLevels = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.currentEvents = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.currentPriceLevels = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.numberOfEventsInBlock = list(np.zeros(len(self.thresholdsForSignalDetector)))
        self.totOscillatorBonus = 0
    def update(self, dcosTraceGenerator, events, closePrice):
        midprice = closePrice.getMid()
        self.totOscillatorBonus = 0
        for j in range(3):
            self.direction[j] = -dcosTraceGenerator[j].mode
            self.dcL[j] = dcosTraceGenerator[j].dcL
            self.osL[j] = dcosTraceGenerator[j].osL
            self.totalMove[j] = dcosTraceGenerator[j].totalMove
            self.extreme[j] = dcosTraceGenerator[j].extreme.level
            self.prev_extremes[j] = dcosTraceGenerator[j].prevExtreme.level
            self.lastDC[j] = dcosTraceGenerator[j].DC.level
            self.textreme[j] = dcosTraceGenerator[j].extreme.time
            self.tprev_extremes[j] = dcosTraceGenerator[j].prevExtreme.time
            self.tlastDC[j] = dcosTraceGenerator[j].DC.time
            self.ref[j] = dcosTraceGenerator[j].reference.level
            self.nos[j] = dcosTraceGenerator[j].nOS
            self.threshold[j] = dcosTraceGenerator[j].threshold
            if abs(events[j]) > 0:
                self.previousEvents[j] = self.currentEvents[j]
                self.previousPriceLevels[j] = self.currentPriceLevels[j]
                self.currentEvents[j] = events[j]
                self.currentPriceLevels[j] = midprice
                self.numberOfEventsInBlock[j] += 1
                #print(f"currentEvents {j} = {self.currentEvents[j]}")
        self.totOscillatorBonus = np.ceil((self.currentEvents[0] + self.currentEvents[1]*2 + self.currentEvents[2]*3) / 3)
        #print(f"midprice = {midprice}, extreme level 0 = {self.extreme[0]}, prev. extreme 0 = {self.prev_extremes[0]}, current price lvl. = {self.currentPriceLevels[0]}, previous price lvl. = {self.previousPriceLevels[0]}")
        return self
    def reset(self, dcosTraceGenerator, closePrice, interpolated_deltas):
        self.numberOfEventsInBlock = list(np.zeros(3))
        for j in range(3):
            #self.currentEvents[j] = dcosTraceGenerator[j].run(closePrice)
            self.direction_tmp[j] = -dcosTraceGenerator[j].mode
            self.tmp_mode[j] = dcosTraceGenerator[j].mode
            self.dcL_tmp[j] = dcosTraceGenerator[j].dcL
            self.osL_tmp[j] = dcosTraceGenerator[j].osL
            self.totalMove_tmp[j] = dcosTraceGenerator[j].totalMove
            self.extreme_tmp[j] = dcosTraceGenerator[j].extreme.level
            self.prev_extremes_tmp[j] = dcosTraceGenerator[j].prevExtreme.level
            self.lastDC_tmp[j] = dcosTraceGenerator[j].DC.level
            self.textreme_tmp[j] = dcosTraceGenerator[j].extreme.time
            self.tprev_extremes_tmp[j] = dcosTraceGenerator[j].prevExtreme.time
            self.tlastDC_tmp[j] = dcosTraceGenerator[j].DC.time
            self.ref_tmp[j] = dcosTraceGenerator[j].reference.level
            self.nos_tmp[j] = dcosTraceGenerator[j].nOS
            # reset
            self.thresholdsForSignalDetector[j] = dcosTraceGenerator[j].threshold =  interpolated_deltas[j]
            dcosTraceGenerator[j] = DcOS_TrendGenerator.DcOS(interpolated_deltas[j], self.tmp_mode[j])
            #self.currentEvents[j] = dcosTraceGenerator[j].run(closePrice)      # !!!
            dcosTraceGenerator[j].mode = self.tmp_mode[j]
            dcosTraceGenerator[j].dcL = self.dcL_tmp[j]
            dcosTraceGenerator[j].osL = self.osL_tmp[j]
            dcosTraceGenerator[j].totalMove = self.totalMove_tmp[j]
            dcosTraceGenerator[j].extreme.level = self.extreme_tmp[j]
            dcosTraceGenerator[j].prevExtreme.level = self.prev_extremes_tmp[j]
            dcosTraceGenerator[j].DC.level = self.lastDC_tmp[j]
            dcosTraceGenerator[j].extreme.time = self.textreme_tmp[j]
            dcosTraceGenerator[j].prevExtreme.time = self.tprev_extremes_tmp[j]
            dcosTraceGenerator[j].DC.time = self.tlastDC_tmp[j]
            dcosTraceGenerator[j].reference.level = self.ref_tmp[j]
            dcosTraceGenerator[j].nOS = self.nos_tmp[j]
        return self, dcosTraceGenerator


class crossSignal:
    __slots__ = ['referencePriceM', 'referencePriceL', 'crossM', 'crossL', 'signal']
    def __init__(self):
        self.crossM = 0
        self.crossL = 0
        self.referencePriceM = 0
        self.referencePriceL = 0
        self.signal = 0
    def generateSignal(self, dcosEvents):
        if (dcosEvents.previousPriceLevels[0]<dcosEvents.currentPriceLevels[1]<dcosEvents.currentPriceLevels[0]) & (dcosEvents.currentPriceLevels[0]!=self.referencePriceM) & (self.crossM < 1):
            self.crossM += 1
            self.referencePriceM = dcosEvents.currentPriceLevels[0]
            #print(f"cross M = {self.crossM}")
        if (dcosEvents.previousPriceLevels[0]>dcosEvents.currentPriceLevels[1]>dcosEvents.currentPriceLevels[0]) & (dcosEvents.currentPriceLevels[0]!=self.referencePriceM) & (self.crossM > -1):
            self.crossM -= 1
            self.referencePriceM = dcosEvents.currentPriceLevels[0]
            #print(f"cross M = {self.crossM}")
        if (dcosEvents.previousPriceLevels[0]<dcosEvents.currentPriceLevels[2]<dcosEvents.currentPriceLevels[0]) & (dcosEvents.currentPriceLevels[0]!=self.referencePriceL) & (self.crossL < 1):
            self.crossL += 1
            self.referencePriceL = dcosEvents.currentPriceLevels[0]
            #print(f"cross L = {self.crossL}")
        if (dcosEvents.previousPriceLevels[0]>dcosEvents.currentPriceLevels[2]>dcosEvents.currentPriceLevels[0]) & (dcosEvents.currentPriceLevels[0]!=self.referencePriceL) & (self.crossL > -1):
            self.crossL -= 1
            self.referencePriceL = dcosEvents.currentPriceLevels[0]
            #print(f"cross L = {self.crossL}")
        if self.crossM + self.crossL == 2:
            self.signal = 1
            self.crossM = 0
            self.crossL = 0
        elif self.crossM + self.crossL == -2:
            self.signal = -1
            self.crossM = 0
            self.crossL = 0
        else:
            self.signal = 0
        #if abs(self.signal) > 0:
            #print(f"cross signal = {self.signal}")
        return self


class trendLineSignal:
    __slots__ = ['overShootDataframe', 'overshootsDirection', 'trendLineDataFrame',
        'estimationIteration', 'estimationTimestamp', 'estimationPoints', 'intercept', 'slope', 'rSquared', # 'model',
        'plotData', 'outputDirImgs', 'signal']
    def __init__(self, overshootsDirection, config_plotData, foldernameImagesSignalDetector):
        self.overShootDataframe = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
        self.overshootsDirection = overshootsDirection
        self.trendLineDataFrame = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
        self.estimationIteration = 0
        self.estimationTimestamp = ''
        self.estimationPoints = 0
        self.intercept = 0
        self.slope = 0
        self.rSquared = 0
        self.signal = 0
        #self.model = []
        self.plotData = config_plotData
        self.outputDirImgs = foldernameImagesSignalDetector
    def updateAndFitToNewData(self, tickReader, LOSevents):
        self.signal = 0
        self.overShootDataframe.loc[len(self.overShootDataframe)] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, LOSevents]
        if len(self.overShootDataframe) == 11:
            self.overShootDataframe = self.overShootDataframe.drop([0])
            self.overShootDataframe.reset_index(inplace=True)
            if 'index' in self.overShootDataframe.columns:
                self.overShootDataframe.drop('index', inplace=True, axis=1)
            if len(self.trendLineDataFrame) > 0:
                tmp_df = self.trendLineDataFrame.copy()
                tmp_df.reset_index(inplace=True)
                if 'index' in self.overShootDataframe.columns:
                    tmp_df.drop('index', inplace=True, axis=1)
                tmp_df.loc[len(tmp_df)] = self.overShootDataframe.iloc[len(self.overShootDataframe)-1]
                subset_df = tmp_df.copy()
                subset_df.set_index('iteration', inplace=True)
                if 'index' in subset_df.columns:
                    subset_df.drop('index', inplace=True, axis=1)
                subset = subset_df.midprice
                #X = np.arange(len(subset)).reshape(-1, 1)
                X = subset.index.values.reshape(-1, 1)
                y = subset.values.reshape(-1, 1)
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r_squared = compute_r_squared(y, y_pred)
                print("")
                print(f"{tickReader.timestamp}: Fitted new observation to support line") #. New R-squared: {np.round(r_squared,3)}, new intercept = {np.round(model.intercept_[0],3)}, new slope = {np.round(model.coef_[0],3)}")
                print(f"self.rSquared - r_squared = {self.rSquared - r_squared}")
                print(f"Last point y = {y[-1]} vs predicted y = {y_pred[-1]}")
                if (self.rSquared - r_squared) > 0.05:
                    #predicted_y = self.alphaParameterExpFunction *  + self.betaParameterExpFunction
                    #y_values_to_find = np.array(self.desiredEventFrequencies) * 100
                    #self.interpolatedThresholds = [find_threshold_for_event_frequency(y, self.alphaParameterExpFunction, self.betaParameterExpFunction) for y in y_values_to_find]
                    if y_pred[-1] > y[-1]:
                    #if tmp_df.midprice.iloc[len(tmp_df)-1] < tmp_df.midprice.iloc[len(tmp_df)-2]:
                        self.signal = -2
                    elif y_pred[-1] < y[-1]:
                    #elif tmp_df.midprice.iloc[len(tmp_df)-1] > tmp_df.midprice.iloc[len(tmp_df)-2]:
                        self.signal = 2
            if abs(self.signal) > 0:
                if self.plotData:
                    plot_X = X[:-1]
                    plot_y = y[:-1]
                    if self.overshootsDirection < 0:
                        prefix = "Support"
                        col = 'green'
                    elif self.overshootsDirection > 0:
                        prefix = "Resistance"
                        col = 'red'
                    plt.scatter(X, y, color='k', label='Data points')
                    plt.plot(plot_X, plot_y, color=col, label='Linear Regression')
                    plt.xlabel('Iteration')
                    plt.ylabel('Midprice')
                    plt.title(f"{prefix} line; model: y={np.round(self.slope,3)}x+{np.round(self.intercept,3)} r-squared={np.round(r_squared,3)}")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f"{self.outputDirImgs}{prefix}_line_deviation_{self.estimationTimestamp}.pdf")
                    plt.show()
                self.overShootDataframe = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
                self.trendLineDataFrame = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
                self.estimationIteration = 0
                self.estimationTimestamp = ''
                self.estimationPoints = 0
                self.intercept = 0
                self.slope = 0
                self.rSquared = 0
                #self.model = []
        return self
    def detectTrendLine(self, tickReader):
        df = self.overShootDataframe.copy()
        #if ((currentEvent>0) & (self.overshootsDirection>0)) | ((currentEvent<0) & (self.overshootsDirection<0))
        if len(df) == 10:
            if self.overshootsDirection < 0:
            #if df.direction.iloc[len(df)-1] < 0:
                df = df[df.direction<0]
            elif self.overshootsDirection > 0:
            #elif df.direction.iloc[len(df)-1] > 0:
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
                    #X = np.arange(len(subset)).reshape(-1, 1)
                    X = subset.index.values.reshape(-1, 1)
                    y = subset.values.reshape(-1, 1)
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    r_squared = compute_r_squared(y, y_pred)
                    if (r_squared>self.rSquared) & (r_squared>0.95): # & (self.supportLineBrakeout==0) & (self.supportLineConfirmation==0):
                        print(f"Trend line updated!")
                        self.trendLineDataFrame = subset_df
                        self.intercept = model.intercept_[0]
                        self.slope = model.coef_[0][0]
                        self.rSquared = r_squared
                        self.estimationPoints = k
                        self.estimationIteration = tickReader.iteration
                        self.estimationTimestamp = tickReader.timestamp
                        #self.model = model
                        if self.plotData:
                            if self.overshootsDirection < 0:
                                prefix = "Support"
                                col = 'green'
                            elif self.overshootsDirection > 0:
                                prefix = "Resistance"
                                col = 'red'
                            plt.scatter(X, y, color='k', label='Data points')
                            plt.plot(X, y_pred, color=col, label='Linear Regression')
                            plt.xlabel('Iteration')
                            plt.ylabel('Midprice')
                            plt.title(f"{prefix} line; k={self.estimationPoints}; model: y={np.round(self.slope,3)}x+{np.round(self.intercept,3)} r-squared={np.round(self.rSquared,3)}")
                            plt.legend()
                            plt.grid(True)
                            plt.savefig(f"{self.outputDirImgs}{prefix}_line_detected_{self.estimationTimestamp}.pdf")
                            plt.show()
        return self


class predictionGenerator:
    __slots__ = ['timeHorizon', 'predictionsDataFrame', 'indicesPredictionsRealised',
                'nrTargetReached', 'nrStopLossReached',
                'saveData', 'colNamesDf', 'outputDir', 'verbose']
    def __init__(self, timeHorizon, columnNamesPredictions, foldernamePredictions, savePredictionData, verbose):
        self.timeHorizon = timeHorizon
        self.predictionsDataFrame =  pd.DataFrame(columns=columnNamesPredictions)
        self.indicesPredictionsRealised = []
        self.saveData = savePredictionData
        self.colNamesDf = columnNamesPredictions
        self.outputDir = foldernamePredictions
        self.verbose = verbose
        self.nrTargetReached = 0
        self.nrStopLossReached = 0
    def generatePrediction(self, signal, threshold, predictionFactor, tickReader):
        if abs(signal) > 0:
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)] = [0, '', 0, 0, 0, 0, 0, 0, 0, '', 0, 0, 0, 0, 0]
            #self.predictionsDataFrame.iterationPredictionStart.loc[len(self.predictionsDataFrame)] = tickReader.iteration
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'iterationPredictionStart'] = tickReader.iteration
            #self.predictionsDataFrame.timestampPredictionStart.loc[len(self.predictionsDataFrame)] = tickReader.timestamp
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'timestampPredictionStart'] = tickReader.timestamp
            #self.predictionsDataFrame.midpricePredictionStart.loc[len(self.predictionsDataFrame)] = tickReader.midprice
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'midpricePredictionStart'] = tickReader.midprice
            #self.predictionsDataFrame.predictionPriceChangePt.loc[len(self.predictionsDataFrame)] = threshold * predictionFactor * 100
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'predictionPriceChangePt'] = threshold * predictionFactor * 100
            if signal < 0:
                predictionDirection = -1
            elif signal > 0:
                predictionDirection = 1
            #self.predictionsDataFrame.predictionDirection.loc[len(self.predictionsDataFrame)] = predictionDirection
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'predictionDirection'] = predictionDirection
            #self.predictionsDataFrame.signal.loc[len(self.predictionsDataFrame)] = signal
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'signal'] = signal
            target = tickReader.midprice+(tickReader.midprice*threshold*predictionFactor*predictionDirection)
            #self.predictionsDataFrame.target.loc[len(self.predictionsDataFrame)] = target
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'target'] = target
            stopLoss = tickReader.midprice-(tickReader.midprice*threshold*predictionFactor*predictionDirection)
            #self.predictionsDataFrame.stopLoss.loc[len(self.predictionsDataFrame)] = stopLoss
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'stopLoss'] = stopLoss
            if self.verbose:
                print("")
                print(f"---------------------------------------------------------")
                print(f"Time timeHorizon: {self.timeHorizon}, {tickReader.timestamp}: New signal = {signal}. Prediction generated!")
                print(f"Entry price = {tickReader.midprice}, Target = {np.round(target,3)}, StopLoss = {np.round(stopLoss,3)} {tickReader.symbol_2}")
        return self
    def checkOngoingPredictions(self, tickReader):
        for i in range(len(self.predictionsDataFrame)):
            if self.predictionsDataFrame.predictionDirection.loc[i] > 0:
                if tickReader.midprice>self.predictionsDataFrame.target.loc[i]:
                    #self.predictionsDataFrame.predictionOutcome.loc[i] = 1
                    #self.predictionsDataFrame.nTargetReached.loc[i] += 1
                    self.predictionsDataFrame.loc[i, 'predictionOutcome'] = 1
                    self.nrTargetReached += 1
                elif tickReader.midprice<self.predictionsDataFrame.stopLoss.loc[i]:
                    #self.predictionsDataFrame.predictionOutcome.loc[i] = -1
                    #self.predictionsDataFrame.self.nStopLossReached.loc[i] += 1
                    self.predictionsDataFrame.loc[i, 'predictionOutcome'] = -1
                    self.nrStopLossReached += 1
            elif self.predictionsDataFrame.predictionDirection.loc[i] < 0:
                if tickReader.midprice<self.predictionsDataFrame.target.loc[i]:
                    #self.predictionsDataFrame.predictionOutcome.loc[i] = 1
                    #self.predictionsDataFrame.self.nTargetReached.loc[i] += 1
                    self.predictionsDataFrame.loc[i, 'predictionOutcome'] = 1
                    self.nrTargetReached += 1
                elif tickReader.midprice>self.predictionsDataFrame.stopLoss.loc[i]:
                    #self.predictionsDataFrame.predictionOutcome.loc[i] = -1
                    #self.predictionsDataFrame.self.nStopLossReached.loc[i] += 1
                    self.predictionsDataFrame.loc[i, 'predictionOutcome'] = -1
                    self.nrStopLossReached += 1
            if abs(self.predictionsDataFrame.predictionOutcome.loc[i]) > 0:
                self.indicesPredictionsRealised.append(i)
                self.predictionsDataFrame.loc[i, 'nrTargetReached'] = self.nrTargetReached
                self.predictionsDataFrame.loc[i, 'nrStopLossReached'] = self.nrStopLossReached
                #self.predictionsDataFrame.iterationPredictionEnd.loc[i] = tickReader.iteration
                #self.predictionsDataFrame.timestampPredictionEnd.loc[i] = tickReader.timestamp
                #self.predictionsDataFrame.midpricePredictionEnd.loc[i] = tickReader.midprice
                #self.predictionsDataFrame.predictionDurationTicks.loc[i] = tickReader.iteration - self.predictionsDataFrame.iterationPredictionStart.loc[i]
                self.predictionsDataFrame.loc[i, 'iterationPredictionEnd'] = tickReader.iteration
                self.predictionsDataFrame.loc[i, 'timestampPredictionEnd'] = tickReader.timestamp
                self.predictionsDataFrame.loc[i, 'midpricePredictionEnd'] = tickReader.midprice
                self.predictionsDataFrame.loc[i, 'predictionDurationTicks'] = tickReader.iteration - self.predictionsDataFrame.iterationPredictionStart.loc[i]
                if self.verbose:
                    print("")
                    print("---------------------------------------------------------")
                    print(f"Time timeHorizon: {self.timeHorizon}, {tickReader.timestamp}: Prediction terminated.")
                    if self.predictionsDataFrame.loc[i, 'predictionOutcome'] == 1:
                        print(f"Target = {np.round(self.predictionsDataFrame.target.loc[i],3)}, stop-loss = {np.round(self.predictionsDataFrame.stopLoss.loc[i],3)}, midprice = {np.round(tickReader.midprice,3)}: TARGET REACHED!")
                    elif self.predictionsDataFrame.loc[i, 'predictionOutcome'] == -1:
                        print(f"Target = {np.round(self.predictionsDataFrame.target.loc[i],3)}, stop-loss = {np.round(self.predictionsDataFrame.stopLoss.loc[i],3)}, midprice = {np.round(tickReader.midprice,3)}: STOP-LOSS REACHED!")
                    print(f"Duration = {self.predictionsDataFrame.loc[i, 'predictionDurationTicks']} s.")
                    print(f"Number of targets reached = {self.predictionsDataFrame.nrTargetReached.loc[i]}; Number of stop-losses reached = {self.predictionsDataFrame.nrStopLossReached.loc[i]}.")
                if self.saveData:
                    dfPredOut = self.predictionsDataFrame.loc[[i]]
                    dfPredOut.to_parquet(f"{self.outputDir}{dfPredOut.timestampPredictionStart.values[0]}_prediction.parquet")
        if len(self.indicesPredictionsRealised) > 0:
            self.predictionsDataFrame.drop(self.indicesPredictionsRealised, inplace=True)
            self.predictionsDataFrame.reset_index(inplace=True)
            if 'index' in self.predictionsDataFrame.columns:
                self.predictionsDataFrame.drop('index', inplace=True, axis=1)
        self.indicesPredictionsRealised = []
        return self



def compute_r_squared(y_true, y_pred):
    residual = y_true - y_pred
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum(residual**2)
    r_squared = 1 - (ss_res / ss_total)
    return r_squared
