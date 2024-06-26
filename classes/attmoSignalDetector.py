import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from DcOS_TrendGenerator import *


class attmoSignalDetector:
    __slots__ = ['timeHorizon', 'thresholdsForSignalDetector',
        'eventDataframe',
        'signalDetected', 'currentForecastLevel',
        'totOscillatorBonus',
        'startingValuesTrendStrength', 'trendForecastLabels',
        'trendStrength', 'trendForecast',
        'alphaParameterExpFunction', 'betaParameterExpFunction',
        'attmoForecastLabels', 'attmoForecast',
        'colNamesDf', 'outputDir',
        'plotData', 'outputDirImgs']
    def __init__(self, config, t, columnNamesSignalDetector, foldernameSignalDetector, foldernameImagesSignalDetector):
        self.timeHorizon = config.timeHorizons[t]
        self.thresholdsForSignalDetector = config.thresholdsForInterpolation[t+20:t+23]
        self.eventDataframe = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'event'])
        self.signalDetected = 0
        self.currentForecastLevel = 0
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

    def updateEventDF(self, tickReader, osEvent):
        self.eventDataframe.loc[len(self.eventDataframe)] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, osEvent]
        if len(self.eventDataframe) == 11:
            self.eventDataframe = self.eventDataframe.drop([0])
            self.eventDataframe.reset_index(inplace=True)
            if 'index' in self.eventDataframe.columns:
                self.eventDataframe.drop('index', inplace=True, axis=1)
        #print("Event added. eventDataframe:")
        #print(self.eventDataframe)
        return self

    def update(self, config, tickReader, dcosTraceGenerator, dcosEventsSignalDetector, predictionGenerator, crossSignal, trendLines, closePrice, iterationBlock, block):
        self.signalDetected = 0

        if block > 0:
            if iterationBlock == 1:
                for j in range(3):
                    dcosTraceGenerator[j] = DcOS_TrendGenerator.DcOS(dcosEventsSignalDetector.thresholdsForSignalDetector[j], dcosEventsSignalDetector[t].direction_tmp[j])
                    dcosTraceGenerator[j].mode = dcosEventsSignalDetector.tmp_mode[j]
                    dcosTraceGenerator[j].dcL = dcosEventsSignalDetector.dcL_tmp[j]
                    dcosTraceGenerator[j].osL = dcosEventsSignalDetector.osL_tmp[j]
                    dcosTraceGenerator[j].totalMove = dcosEventsSignalDetector.totalMove_tmp[j]
                    dcosTraceGenerator[j].extreme.level = dcosEventsSignalDetector.extreme_tmp[j]
                    dcosTraceGenerator[j].prevExtreme.level = dcosEventsSignalDetector.prev_extremes_tmp[j]
                    dcosTraceGenerator[j].DC.level = dcosEventsSignalDetector.lastDC_tmp[j]
                    dcosTraceGenerator[j].extreme.time = dcosEventsSignalDetector.textreme_tmp[j]
                    dcosTraceGenerator[j].prevExtreme.time = dcosEventsSignalDetector.tprev_extremes_tmp[j]
                    dcosTraceGenerator[j].DC.time = dcosEventsSignalDetector.tlastDC_tmp[j]
                    dcosTraceGenerator[j].reference.level = dcosEventsSignalDetector.ref_tmp[j]
                    dcosTraceGenerator[j].nOS = dcosEventsSignalDetector[t].nos_tmp[j]

        events = list(np.zeros(3))
        for j in range(3):
            events[j] = dcosTraceGenerator[j].run(closePrice)
        dcosEventsSignalDetector = dcosEventsSignalDetector.update(dcosTraceGenerator, events, closePrice)

        if block > 0:
            if len(predictionGenerator.predictionsDataFrame) > 0:
                predictionGenerator = predictionGenerator.checkOngoingPredictions(tickReader)

            if abs(events[2]) > 0:
                self = self.updateEventDF(tickReader, events[2])
                if abs(events[2]) == 2:
                    for i in range(2):
                        trendLines[i] = trendLines[i].updateAndFitToNewData(tickReader, events[2])

                    if (trendLines[0].signal==2) | (trendLines[1].signal==2):
                        self.signalDetected = 2
                    elif (trendLines[0].signal==-2) | (trendLines[1].signal==-2):
                        self.signalDetected = -2
                    elif (trendLines[0].signal==1) | (trendLines[1].signal==1):
                        self.signalDetected = 1
                    elif (trendLines[0].signal==-1) | (trendLines[1].signal==-1):
                        self.signalDetected = -1

                    if (abs(trendLines[0].signal)>0) & (trendLines[1].estimationPoints>0):
                        if trendLines[0].signal > 0:
                            self.signalDetected = 3
                        else:
                            self.signalDetected = -3
                    elif ((abs(trendLines[1].signal)>0 & trendLines[0].estimationPoints>0)):
                        if trendLines[1].signal > 0:
                            self.signalDetected = 3
                        else:
                            self.signalDetected = -3

                    if events[2] == -2:
                        trendLines[0] = trendLines[0].detectTrendLine(self.eventDataframe)
                    elif events[2] == 2:
                        trendLines[1] = trendLines[1].detectTrendLine(self.eventDataframe)

                    if len(predictionGenerator.predictionsDataFrame) == 0:
                        maxSignalPre = 0
                    else:
                        maxSignalPre = np.max(abs(np.array(predictionGenerator.predictionsDataFrame.signal)))
                    predictionGenerator = predictionGenerator.generatePrediction(self.signalDetected, dcosTraceGenerator[2].threshold, config.predictionFactor, tickReader)

                    if len(predictionGenerator.predictionsDataFrame) > 0:
                        if abs(self.signalDetected) > maxSignalPre:
                            print("")
                            print(f"---------------------------------------------------------")
                            print(f"Time timeHorizon: {self.timeHorizon}, {tickReader.timestamp}: Current ATTMO forecast: {self.currentForecastLevel} New ATTMO forecast: {self.signalDetected}")
                            self.currentForecastLevel = self.signalDetected
                        elif (self.signalDetected!=0) & (abs(self.signalDetected)==maxSignalPre):
                            idxMaxPosSignal = np.where(predictionGenerator.predictionsDataFrame.signal == maxSignalPre)
                            idxMaxNegSignal = np.where(predictionGenerator.predictionsDataFrame.signal == -maxSignalPre)
                            if (self.signalDetected>0) & (len(idxMaxPosSignal[0])>len(idxMaxNegSignal[0])):
                                print("")
                                print(f"---------------------------------------------------------")
                                print(f"Time timeHorizon: {self.timeHorizon}, {tickReader.timestamp}: Current ATTMO forecast: {self.currentForecastLevel} New ATTMO forecast: {self.signalDetected}")
                                self.currentForecastLevel = self.signalDetected
                            elif (self.signalDetected<0) & (len(idxMaxPosSignal[0])<len(idxMaxNegSignal[0])):
                                print("")
                                print(f"---------------------------------------------------------")
                                print(f"Time timeHorizon: {self.timeHorizon}, {tickReader.timestamp}: Current ATTMO forecast: {self.currentForecastLevel} New ATTMO forecast: {self.signalDetected}")
                                self.currentForecastLevel = self.signalDetected
                        self.trendStrength = self.startingValuesTrendStrength[self.currentForecastLevel + 3]
                        self.trendStrength += dcosEventsSignalDetector.totOscillatorBonus
                        if self.trendStrength > 100:
                            self.trendStrength = 100
                        elif self.trendStrength < 1:
                            self.trendStrength = 1
                        self.trendForecast = self.trendForecastLabels[int(np.floor(self.trendStrength/10))]
                        self.attmoForecast = self.attmoForecastLabels[self.currentForecastLevel + 3]

        if config.saveSignalDetectionData:
            self.saveSignalDetection(tickReader, dcosEventsSignalDetector, events, iterationBlock, block, trendLines)
        return self


    def saveSignalDetection(self, tickReader, dcosEventsSignalDetector, events, iterationBlock, block, trendLines):
        signalDetector_core = [(self.thresholdsForSignalDetector[i], events[i], dcosEventsSignalDetector.numberOfEventsInBlock[i]) for i in range(len(self.thresholdsForSignalDetector))]
        df_signalDetector_core = [item for t in signalDetector_core for item in t]
        df_signalDetector = pd.DataFrame(columns=self.colNamesDf)
        df_signalDetector.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, iterationBlock, block] + df_signalDetector_core + [self.signalDetected, self.currentForecastLevel, self.trendStrength, self.trendForecast, self.attmoForecast, trendLines[0].intercept, trendLines[0].slope, trendLines[0].rSquared, trendLines[0].estimationPoints, trendLines[0].iterationFirstSample, trendLines[0].timestampFirstSample, trendLines[0].midpriceFirstSample, trendLines[0].iterationLastSample, trendLines[0].timestampLastSample, trendLines[0].midpriceLastSample, trendLines[1].intercept, trendLines[1].slope, trendLines[1].rSquared, trendLines[1].estimationPoints, trendLines[1].iterationFirstSample, trendLines[1].timestampFirstSample, trendLines[1].midpriceFirstSample, trendLines[1].iterationLastSample, trendLines[1].timestampLastSample, trendLines[1].midpriceLastSample]
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
            #dcosTraceGenerator[j] = DcOS_TrendGenerator.DcOS(interpolated_deltas[j], self.tmp_mode[j])
            #res = dcosTraceGenerator[j].run(closePrice)      # !!!
            #dcosTraceGenerator[j].mode = self.tmp_mode[j]
            #dcosTraceGenerator[j].dcL = self.dcL_tmp[j]
            #dcosTraceGenerator[j].osL = self.osL_tmp[j]
            #dcosTraceGenerator[j].totalMove = self.totalMove_tmp[j]
            #dcosTraceGenerator[j].extreme.level = self.extreme_tmp[j]
            #dcosTraceGenerator[j].prevExtreme.level = self.prev_extremes_tmp[j]
            #dcosTraceGenerator[j].DC.level = self.lastDC_tmp[j]
            #dcosTraceGenerator[j].extreme.time = self.textreme_tmp[j]
            #dcosTraceGenerator[j].prevExtreme.time = self.tprev_extremes_tmp[j]
            #dcosTraceGenerator[j].DC.time = self.tlastDC_tmp[j]
            #dcosTraceGenerator[j].reference.level = self.ref_tmp[j]
            #dcosTraceGenerator[j].nOS = self.nos_tmp[j]
        return self


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
        if (dcosEvents.previousPriceLevels[0]>dcosEvents.currentPriceLevels[1]>dcosEvents.currentPriceLevels[0]) & (dcosEvents.currentPriceLevels[0]!=self.referencePriceM) & (self.crossM > -1):
            self.crossM -= 1
            self.referencePriceM = dcosEvents.currentPriceLevels[0]
        if (dcosEvents.previousPriceLevels[0]<dcosEvents.currentPriceLevels[2]<dcosEvents.currentPriceLevels[0]) & (dcosEvents.currentPriceLevels[0]!=self.referencePriceL) & (self.crossL < 1):
            self.crossL += 1
            self.referencePriceL = dcosEvents.currentPriceLevels[0]
        if (dcosEvents.previousPriceLevels[0]>dcosEvents.currentPriceLevels[2]>dcosEvents.currentPriceLevels[0]) & (dcosEvents.currentPriceLevels[0]!=self.referencePriceL) & (self.crossL > -1):
            self.crossL -= 1
            self.referencePriceL = dcosEvents.currentPriceLevels[0]
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
        return self


class trendLineSignal:
    __slots__ = ['overShootDataframe', 'overshootsDirection', 'trendLineDataFrame', 'X', 'y',
        'estimationPoints', 'intercept', 'slope', 'rSquared',
        'iterationFirstSample', 'timestampFirstSample', 'midpriceFirstSample',
        'iterationLastSample', 'timestampLastSample', 'midpriceLastSample',
        'plotData', 'outputDirImgs', 'signal']
    def __init__(self, overshootsDirection, config_plotData, foldernameImagesSignalDetector):
        self.overShootDataframe = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
        self.X = []
        self.y = []
        self.overshootsDirection = overshootsDirection
        self.trendLineDataFrame = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
        #self.estimationIteration = 0
        #self.estimationTimestamp = ''
        self.estimationPoints = 0
        self.intercept = 0
        self.slope = 0
        self.rSquared = 0
        self.signal = 0
        self.iterationFirstSample = 0
        self.timestampFirstSample = ''
        self.midpriceFirstSample = 0
        self.iterationLastSample = 0
        self.timestampLastSample = ''
        self.midpriceLastSample = 0
        #self.updateModelWithNewOS = 0
        self.plotData = config_plotData
        self.outputDirImgs = foldernameImagesSignalDetector
    def updateAndFitToNewData(self, tickReader, osEvent):
        self.signal = 0
        self.overShootDataframe.loc[len(self.overShootDataframe)] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, osEvent]
        if len(self.overShootDataframe) == 11:
            self.overShootDataframe = self.overShootDataframe.drop([0])
            self.overShootDataframe.reset_index(inplace=True)
            if 'index' in self.overShootDataframe.columns:
                self.overShootDataframe.drop('index', inplace=True, axis=1)
            if len(self.trendLineDataFrame) > 0:
                lX = list(self.X) + [tickReader.iteration]
                X = np.array(lX).reshape(-1, 1)
                ly = list(self.y) + [tickReader.midprice]
                y = np.array(ly).reshape(-1, 1)
                y_pred = []
                [y_pred.append(self.slope*x+self.intercept) for x in X]
                r_squared = compute_r_squared(y, y_pred)
                if (self.rSquared - r_squared) > 0.1:
                    if y_pred[-1] > y[-1]:
                        if self.estimationPoints < 4:
                            self.signal = -1
                        else:
                            self.signal = -2
                    elif y_pred[-1] < y[-1]:
                        if self.estimationPoints < 4:
                            self.signal = 1
                        else:
                            self.signal = 2
                    if self.plotData:
                        self.plotConfirmationOrBrakeout(X, y, y_pred)
                    self = self.reset()
        return self
    def detectTrendLine(self, eventDataframe):
        df = self.overShootDataframe.copy()
        if len(df) == 10:
            if self.overshootsDirection < 0:
                df = df[df.direction<0]
            elif self.overshootsDirection > 0:
                df = df[df.direction>0]
            if len(df) > 2:
                for k in range(3,len(df)):
                    self = self.fitLineToNewSet(df, k, eventDataframe)
        return self
    def fitLineToSetConfirmation(self, df, k):
        setToFit_df = df.iloc[len(df)-k:len(df)]
        setToFit_df.set_index('iteration', inplace=True)
        if 'index' in setToFit_df.columns:
            setToFit_df.drop('index', inplace=True, axis=1)
        setToFit = setToFit_df.midprice
        X = setToFit.index.values.reshape(-1, 1)
        y = setToFit.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        #print(f"Trend line updated!")
        self.trendLineDataFrame = setToFit_df
        self.X = setToFit.index.values
        self.y = setToFit.values
        self.intercept = model.intercept_[0]
        self.slope = model.coef_[0][0]
        self.rSquared = compute_r_squared(y, y_pred)
        self.estimationPoints = k
        if self.plotData:
            self.plotNewTrendLine(X, y, y_pred)
        return self
    def fitLineToNewSet(self, df, k, eventDataframe):
        setToFit_df = df.iloc[len(df)-k:len(df)]
        setToFit_df.set_index('iteration', inplace=True)
        if 'index' in setToFit_df.columns:
            setToFit_df.drop('index', inplace=True, axis=1)
        setToFit = setToFit_df.midprice

        event_df = eventDataframe.iloc[len(eventDataframe)-k:len(eventDataframe)]
        event_df.set_index('iteration', inplace=True)
        if 'index' in event_df.columns:
            event_df.drop('index', inplace=True, axis=1)
        eventMid = event_df.midprice

        if list(setToFit.index.values) != list(eventMid.index.values):
            X = setToFit.index.values.reshape(-1, 1)
            y = setToFit.values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r_squared = compute_r_squared(y, y_pred)
            if (r_squared>self.rSquared) & (r_squared>0.95):
                #print(f"Trend line updated!")
                self.trendLineDataFrame = setToFit_df
                self.X = setToFit.index.values
                self.y = setToFit.values
                self.intercept = model.intercept_[0]
                self.slope = model.coef_[0][0]
                self.rSquared = r_squared
                self.estimationPoints = k
                self.iterationFirstSample = setToFit_df.index.values[0]
                self.timestampFirstSample = setToFit_df.timestamp.iloc[0]
                self.midpriceFirstSample = setToFit_df.midprice.iloc[0]
                self.iterationLastSample = setToFit_df.index.values[len(setToFit_df)-1]
                self.timestampLastSample = setToFit_df.timestamp.iloc[len(setToFit_df)-1]
                self.midpriceLastSample = setToFit_df.midprice.iloc[len(setToFit_df)-1]
                if self.plotData:
                    self.plotNewTrendLine(X, y, y_pred)
        return self
    def plotNewTrendLine(self, X, y, y_pred):
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
        plt.savefig(f"{self.outputDirImgs}{prefix}_line_detected_{self.timestampFirstSample}.pdf") # estimationTimestamp
        plt.show()
    def plotConfirmationOrBrakeout(self, X, y, y_pred):
        if self.overshootsDirection < 0:
            prefix = "Support"
            col = 'green'
            plt.plot(X[-2:], y_pred[-2:], '--g')
        elif self.overshootsDirection > 0:
            prefix = "Resistance"
            col = 'red'
            plt.plot(X[-2:], y_pred[-2:], '--r')
        plt.scatter(X, y, color='k', label='Data points')
        plt.plot(X[:-1], y_pred[:-1], color=col, label='Linear Regression')
        plt.xlabel('Iteration')
        plt.ylabel('Midprice')
        plt.title(f"{prefix} line; model: y={np.round(self.slope,3)}x+{np.round(self.intercept,3)} r-squared={np.round(self.rSquared,3)}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.outputDirImgs}{prefix}_line_deviation_{self.timestampFirstSample}.pdf") # estimationTimestamp
        plt.show()
    def reset(self):
        #self.overShootDataframe = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
        self.X = []
        self.y = []
        self.trendLineDataFrame = pd.DataFrame(columns = ['iteration', 'timestamp', 'midprice', 'direction'])
        self.estimationPoints = 0
        self.intercept = 0
        self.slope = 0
        self.rSquared = 0
        self.iterationFirstSample = 0
        self.timestampFirstSample = ''
        self.midpriceFirstSample = 0
        self.iterationLastSample = 0
        self.timestampLastSample = ''
        self.midpriceLastSample = 0
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
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'iterationPredictionStart'] = tickReader.iteration
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'timestampPredictionStart'] = tickReader.timestamp
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'midpricePredictionStart'] = tickReader.midprice
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'predictionPriceChangePt'] = threshold * predictionFactor * 100
            if signal < 0:
                predictionDirection = -1
            elif signal > 0:
                predictionDirection = 1
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'predictionDirection'] = predictionDirection
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'signal'] = signal
            target = tickReader.midprice+(tickReader.midprice*threshold*predictionFactor*predictionDirection)
            self.predictionsDataFrame.loc[len(self.predictionsDataFrame)-1, 'target'] = target
            stopLoss = tickReader.midprice-(tickReader.midprice*threshold*predictionFactor*predictionDirection)
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
                    self.predictionsDataFrame.loc[i, 'predictionOutcome'] = 1
                    self.nrTargetReached += 1
                elif tickReader.midprice<self.predictionsDataFrame.stopLoss.loc[i]:
                    self.predictionsDataFrame.loc[i, 'predictionOutcome'] = -1
                    self.nrStopLossReached += 1
            elif self.predictionsDataFrame.predictionDirection.loc[i] < 0:
                if tickReader.midprice<self.predictionsDataFrame.target.loc[i]:
                    self.predictionsDataFrame.loc[i, 'predictionOutcome'] = 1
                    self.nrTargetReached += 1
                elif tickReader.midprice>self.predictionsDataFrame.stopLoss.loc[i]:
                    self.predictionsDataFrame.loc[i, 'predictionOutcome'] = -1
                    self.nrStopLossReached += 1
            if abs(self.predictionsDataFrame.predictionOutcome.loc[i]) > 0:
                self.indicesPredictionsRealised.append(i)
                self.predictionsDataFrame.loc[i, 'nrTargetReached'] = self.nrTargetReached
                self.predictionsDataFrame.loc[i, 'nrStopLossReached'] = self.nrStopLossReached
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
    if r_squared < 0:
        r_squared = 0
    elif r_squared > 1:
        r_squared = 1
    return r_squared
