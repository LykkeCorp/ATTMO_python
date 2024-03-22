import numpy as np
import pandas as pd
from DcOS_TrendGenerator import *


class ATTMO_signal_detector:
    __slots__ = ['timeHorizon', 'thresholdsForSignalDetector',
        'currentEventsSignalDetector', 'currentLevelsSignalDetector',
        'previousEventsSignalDetector', 'previousLevelsSignalDetector',
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
        'totOscillatorBonus', 'currentSignalLevel',
        'startingValuesTrendStrength', 'trendForecastLabels',
        'trendStrength', 'trendForecast']
    def __init__(self, config, t):
        self.timeHorizon = config.timeHorizons[t]
        self.thresholdsForSignalDetector = list(np.zeros(3))
        self.currentEventsSignalDetector = list(np.zeros(3))
        self.currentLevelsSignalDetector = list(np.zeros(3))
        self.previousEventsSignalDetector = list(np.zeros(3))
        self.previousLevelsSignalDetector = list(np.zeros(3))
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
        #self.signalDetected = 0
        self.currentSignalLevel = 0
        self.startingValuesTrendStrength = config.startingValuesTrendStrength
        self.trendForecastLabels = config.trendForecastLabels
        self.trendStrength = 51
        self.trendForecast = "NONE"
    def run(self, config, t, tickReader, dcosSignalDetector, closePrice, predictionIsOngoing, iterationBlock, block, colNamesSigDetect, foldernameSignalDetector):
        midprice = closePrice.getMid()
        tempEvents = list(np.zeros(3))
        #self.signalDetected = 0
        for j in range(3):
            tempEvents[j] = dcosSignalDetector[j].run(closePrice)
            if abs(tempEvents[j]) > 0:
                self.previousEventsSignalDetector[j] = self.currentEventsSignalDetector[j]
                self.previousLevelsSignalDetector[j] = self.currentLevelsSignalDetector[j]
                self.currentEventsSignalDetector[j] = tempEvents[j]
                self.currentLevelsSignalDetector[j] = midprice
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
        self.totOscillatorBonus = np.round((self.eventsOscillators[0] + (self.eventsOscillators[1]*2) + (self.eventsOscillators[2]*3)) / 10)
        if (self.previousLevelsSignalDetector[0]<self.currentLevelsSignalDetector[1]<self.currentLevelsSignalDetector[0]) & (self.previousLevelsSignalDetector[1]<self.currentLevelsSignalDetector[2]<self.currentLevelsSignalDetector[1]) & (block>0) & (predictionIsOngoing==0):
            self.currentSignalLevel = 3
            self.trendStrength = self.startingValuesTrendStrength[5] + self.totOscillatorBonus
        elif (self.previousLevelsSignalDetector[0]>self.currentLevelsSignalDetector[1]>self.currentLevelsSignalDetector[0]) & (self.previousLevelsSignalDetector[1]>self.currentLevelsSignalDetector[2]>self.currentLevelsSignalDetector[1]) & (block>0) & (predictionIsOngoing==0):
            self.currentSignalLevel = -3
            self.trendStrength = self.startingValuesTrendStrength[0] + self.totOscillatorBonus
        if self.currentSignalLevel < 3:
            if (self.previousLevelsSignalDetector[0]<=self.currentLevelsSignalDetector[1]<=self.currentLevelsSignalDetector[0]) & (self.previousLevelsSignalDetector[1]<self.currentLevelsSignalDetector[2]<self.currentLevelsSignalDetector[1]) & (block>0) & (predictionIsOngoing==0):
                self.currentSignalLevel = 2
                self.trendStrength = self.startingValuesTrendStrength[4] + self.totOscillatorBonus
            elif (self.previousLevelsSignalDetector[0]>=self.currentLevelsSignalDetector[1]>=self.currentLevelsSignalDetector[0]) & (self.previousLevelsSignalDetector[1]>self.currentLevelsSignalDetector[2]>self.currentLevelsSignalDetector[1]) & (block>0) & (predictionIsOngoing==0):
                self.currentSignalLevel = -2
                self.trendStrength = self.startingValuesTrendStrength[1] + self.totOscillatorBonus
        if self.currentSignalLevel < 2:
            if (self.previousLevelsSignalDetector[0]<self.currentLevelsSignalDetector[1]<self.currentLevelsSignalDetector[0]) & (self.previousLevelsSignalDetector[1]<=self.currentLevelsSignalDetector[2]<=self.currentLevelsSignalDetector[1]) & (block>0) & (predictionIsOngoing==0):
                self.currentSignalLevel = 1
                self.trendStrength = self.startingValuesTrendStrength[3] + self.totOscillatorBonus
            elif (self.previousLevelsSignalDetector[0]>self.currentLevelsSignalDetector[1]>self.currentLevelsSignalDetector[0]) & (self.previousLevelsSignalDetector[1]>=self.currentLevelsSignalDetector[2]>=self.currentLevelsSignalDetector[1]) & (block>0) & (predictionIsOngoing==0):
                self.currentSignalLevel = -1
                self.trendStrength = self.startingValuesTrendStrength[2] + self.totOscillatorBonus
        if self.currentSignalLevel == 0:
            self.trendStrength = 50 + self.totOscillatorBonus
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
        self.currentEventsSignalDetector = list(np.zeros(3))
        signalDetector_core = [(self.thresholdsForSignalDetector[i], self.currentEventsSignalDetector[i], self.numberOfEventsInBlock[i]) for i in range(len(dcosSignalDetector))]
        df_signalDetector_core = [item for t in signalDetector_core for item in t]
        df_signalDetector = pd.DataFrame(columns=colNamesSigDetect)
        df_signalDetector.loc[0] = [tickReader.iteration, tickReader.timestamp, tickReader.midprice, iterationBlock, block] + df_signalDetector_core + [self.currentSignalLevel, self.trendStrength, self.trendForecast]
        df_signalDetector.to_parquet(f"{foldernameSignalDetector}{tickReader.timestamp}_signalDetector.parquet")
        if (config.verbose) & (iterationBlock%60==1):
            print(f"Time timeHorizon: {config.timeHorizons[t]}: trend strength = {self.trendStrength}, trend forecast = {self.trendForecast}")
        return self
    def reset(self, dcosSignalDetector, closePrice, interpolated_deltas):
        self.numberOfEventsInBlock = list(np.zeros(3))
        self.currentSignalLevel = 0
        self.trendStrength = 51
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
