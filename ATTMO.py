import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import json

import websocket
from binance.client import Client

from DcOS_TrendGenerator import *
from functions import *

from classes.ATTMO_config import ATTMO_config
config = ATTMO_config()

from classes.ATTMO_tick_reader import ATTMO_tick_reader
from classes.ATTMO_interpolator import ATTMO_interpolator
from classes.ATTMO_signal_detector import ATTMO_signal_detector
from classes.ATTMO_prediction_generator import ATTMO_prediction_generator

if (config.runOnNotebook) & (config.clearOutput):
    from IPython.display import clear_output


####### declare symbol #######
assetString = f"{config.symbol_1.lower()}{config.symbol_2.lower()}@kline_1s"


####### create folders #######
now = datetime.now()
if config.runOnLocal:
    foldername = f"{config.symbol_1}{config.symbol_2}_{now.strftime('%d-%m-%Y_%H-%M-%S/')}"
else:
    foldername = f"/data/{config.symbol_1}{config.symbol_2}_{now.strftime('%d-%m-%Y_%H-%M-%S/')}"
foldernameTicks = foldername+"ticks/"
functions.create_folders(foldername, config.timeHorizons)


####### set dataframes column names #######
columnNamesTickReader, columnNamesInterpolator, columnNamesSignalDetector, columnNamesPredictor = functions.generate_dataframes_column_names()


####### save copy of configuration file #######
functions.copy_configuration_file(foldername, "ATTMO_config.txt", now.strftime('%d-%m-%Y_%H-%M-%S'), config.symbol_1, config.symbol_2)


####### initialise tick reader #######
tickReader = ATTMO_tick_reader(config)


####### initialise classes containers #######
interpolators = [[] for _ in range(len(config.desiredEventFrequenciesList))]
dcosInterpolation = [[] for _ in range(len(config.desiredEventFrequenciesList))]
signalDetectors = [[] for _ in range(len(config.desiredEventFrequenciesList))]
dcosSignalDetection = [[] for _ in range(len(config.desiredEventFrequenciesList))]
predictionGenerators = [[] for _ in range(len(config.desiredEventFrequenciesList))]


for t in range(len(config.desiredEventFrequenciesList)):
    ####### initialise interpolators #######
    interpolators[t] = ATTMO_interpolator(config, t)
    dcosInterpolation[t] = [[] for _ in range(len(config.thresholdsForInterpolation))]
    for i in range(len(config.thresholdsForInterpolation)):
        dcosInterpolation[t][i] = DcOS_TrendGenerator.DcOS(config.thresholdsForInterpolation[i], -1)


    ####### initialise signal detectors #######
    signalDetectors[t] = ATTMO_signal_detector(config, t)
    dcosSignalDetection[t] = [[] for _ in range(3)]
    for j in range(3):
        dcosSignalDetection[t][j] = DcOS_TrendGenerator.DcOS(interpolators[t].interpolatedThresholds[j], -1)


    ####### initialise predictors #######
    predictionGenerators[t] = ATTMO_prediction_generator(config, t)


def on_open(ws):
    print("Connection is open.")


def on_close(ws, status, message):
    print("Connection is closed.")


####### main loop #######
def on_message(ws, message):
    global closePrice, midprice, price_1, df
    global client, config, assetString
    global tickReader, interpolators, signalDetectors, predictionGenerators, dcosInterpolation, dcosSignalDetection
    global foldername, foldernameTicks
    global columnNamesTickReader, columnNamesInterpolator, columnNamesSignalDetector, columnNamesPredictor


    ####### 1. Read tick & update counters #######
    message = json.loads(message)
    df, midprice = functions.manipulation(message)
    tickReader = tickReader.run(df.timestamp[0], midprice, config.saveTickData, columnNamesTickReader, foldernameTicks)
    closePrice = DcOS_TrendGenerator.Price(tickReader.midprice, tickReader.midprice, tickReader.iteration)


    for t in range(len(config.timeHorizons)):
        foldernameTimeHorizon = foldername+config.timeHorizons[t]+"/"
        foldernameInterpolation = foldernameTimeHorizon+"interpolation/"
        foldernameSignalDetector = foldernameTimeHorizon+"signal_detector/"
        foldernamePredictions = foldernameTimeHorizon+"predictions/"


        ####### 2. run interpolation #######
        interpolators[t] = interpolators[t].run(t, dcosInterpolation[t], closePrice)


        ####### 3. run signal detection #######
        signalDetectors[t] = signalDetectors[t].run(config, t, tickReader, dcosSignalDetection[t], closePrice, predictionGenerators[t].predictionIsOngoing, interpolators[t].iterationBlock, interpolators[t].block, columnNamesSignalDetector, foldernameSignalDetector)


        ####### 4. generate prediction #######
        predictionGenerators[t] = predictionGenerators[t].run(tickReader, interpolators[t].iterationBlock, interpolators[t].block, interpolators[t].powerLawParameters, signalDetectors[t].thresholdsForSignalDetector[1], signalDetectors[t].currentSignalLevel, config, columnNamesPredictor, foldername, foldernamePredictions)


        ####### 5. compute volatility #######
        if interpolators[t].iterationBlock == config.blockLengths[t]:
            if config.clearOutput:
                if config.runOnNotebook:
                    clear_output(wait=False)
                else:
                    if config.runOnLocal:
                        os.system("cls")
                    else:
                        os.system("clear")
            interpolators[t] = interpolators[t].fit_power_law(config, tickReader, columnNamesInterpolator, foldernameInterpolation)
            signalDetectors[t], dcosSignalDetection[t] = signalDetectors[t].reset(dcosSignalDetection[t], closePrice, interpolators[t].interpolatedThresholds)



####### Run Program #######
def connect_to_binance_stream():
    socket = "wss://stream.binance.com:9443/stream?streams="+assetString
    ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
    ws.run_forever()


####### Initial connection #######
connect_to_binance_stream()
