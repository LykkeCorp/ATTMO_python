import numpy as np
import pandas as pd
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
        'timePremiumPredictionOn', 'timePremiumPredictionOff']
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
    def run(self, config, t, tickReader, dcosSignalDetector, closePrice, windLevel, iterationBlock, block, columnNamesSignalDetector, foldernameSignalDetector, columnNamesPredictionGenerated, foldernamePredictionGenerated, columnNamesPredictionOutcome, foldernamePredictionOutcome):
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
            self.totOscillatorBonus = np.ceil((self.eventsOscillators[0] + (self.eventsOscillators[1]*2) + (self.eventsOscillators[2]*3)) / 10)
        if block > 0:
            if (self.previousPriceLevelsSignalDetector[0]<self.currentPriceLevelsSignalDetector[1]<self.currentPriceLevelsSignalDetector[0]) & (self.previousPriceLevelsSignalDetector[0]<self.currentPriceLevelsSignalDetector[2]<self.currentPriceLevelsSignalDetector[0]) & (block>0) & (self.premiumPredictionIsOngoing==0):
                if self.ongoingSignalLevel != 3:
                    self.signalDetected = 3
                    self.totOscillatorBonus = 0
                self.ongoingSignalLevel = 3
                self.trendStrength = self.startingValuesTrendStrength[5]
                self.attmoForecast = self.attmoForecastLabels[5]
            elif (self.previousPriceLevelsSignalDetector[0]>self.currentPriceLevelsSignalDetector[1]>self.currentPriceLevelsSignalDetector[0]) & (self.previousPriceLevelsSignalDetector[0]>self.currentPriceLevelsSignalDetector[2]>self.currentPriceLevelsSignalDetector[0]) & (block>0) & (self.premiumPredictionIsOngoing==0):
                if self.ongoingSignalLevel != -3:
                    self.signalDetected = -3
                    self.totOscillatorBonus = 0
                self.ongoingSignalLevel = -3
                self.trendStrength = self.startingValuesTrendStrength[0]
                self.attmoForecast = self.attmoForecastLabels[0]
            if abs(self.ongoingSignalLevel) < 3:
                if (self.previousPriceLevelsSignalDetector[0]<self.currentPriceLevelsSignalDetector[2]<self.currentPriceLevelsSignalDetector[0]) & (block>0):
                    if self.ongoingSignalLevel != 2:
                        self.signalDetected = 2
                        self.totOscillatorBonus = 0
                    self.ongoingSignalLevel = 2
                    self.trendStrength = self.startingValuesTrendStrength[4]
                    self.attmoForecast = self.attmoForecastLabels[4]
                elif (self.previousPriceLevelsSignalDetector[0]>self.currentPriceLevelsSignalDetector[2]>self.currentPriceLevelsSignalDetector[0]) & (block>0):
                    if self.ongoingSignalLevel != -2:
                        self.signalDetected = -2
                        self.totOscillatorBonus = 0
                    self.ongoingSignalLevel = -2
                    self.trendStrength = self.startingValuesTrendStrength[1]
                    self.attmoForecast = self.attmoForecastLabels[1]
            if abs(self.ongoingSignalLevel) < 2:
                if tempEvents[2] == 1:
                    if self.ongoingSignalLevel != 1:
                        self.signalDetected = 1
                        #self.totOscillatorBonus = 0
                    self.ongoingSignalLevel = 1
                    self.trendStrength = self.startingValuesTrendStrength[3]
                    self.attmoForecast = self.attmoForecastLabels[3]
                elif tempEvents[2] == -1:
                    if self.ongoingSignalLevel != -1:
                        self.signalDetected = -1
                        #self.totOscillatorBonus = 0
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
        signalDetector_core = [(self.thresholdsForSignalDetector[i], self.currentEventsSignalDetector[i], self.numberOfEventsInBlock[i]) for i in range(len(dcosSignalDetector))]
        df_signalDetector_core = [item for t in signalDetector_core for item in t]
        df_signalDetector = pd.DataFrame(columns=columnNamesSignalDetector)
        df_signalDetector.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, iterationBlock, block] + df_signalDetector_core + [self.signalDetected, self.ongoingSignalLevel, self.trendStrength, self.trendForecast, self.attmoForecast]
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
