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

from classes.attmoConfig import attmoConfig
from classes.attmoTickReader import attmoTickReader
from classes.attmoInterpolator import attmoInterpolator
from classes.attmoSignalDetector import attmoSignalDetector
from classes.attmoSignalDetector import intrinsicTimeEventsSignalDetector
from classes.attmoSignalDetector import crossSignal
from classes.attmoSignalDetector import trendLineSignal
from classes.attmoSignalDetector import predictionGenerator


config = attmoConfig()


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
columnNamesTickReader, columnNamesInterpolator, columnNamesSignalDetector, columnNamesPredictions = functions.generate_dataframes_column_names()


####### save copy of configuration file #######
functions.copy_configuration_file(foldername, "ATTMO_config.txt", now.strftime('%d-%m-%Y_%H-%M-%S'), config.symbol_1, config.symbol_2)


####### initialise tick reader #######
tickReader = attmoTickReader(config, columnNamesTickReader, foldernameTicks)


####### initialise classes containers #######
interpolators = [[] for _ in range(len(config.timeHorizons))]
dcosInterpolation = [[] for _ in range(len(config.timeHorizons))]
signalDetectors = [[] for _ in range(len(config.timeHorizons))]
eventsSignalDetector = [[] for _ in range(len(config.timeHorizons))]
dcosSignalDetection = [[] for _ in range(len(config.timeHorizons))]
crossSignals = [[] for _ in range(len(config.timeHorizons))]
trendLineSignals = [[] for _ in range(len(config.timeHorizons))]
predictionGenerators = [[] for _ in range(len(config.timeHorizons))]


for t in range(len(config.timeHorizons)):
    foldernameTimeHorizon = foldername+config.timeHorizons[t]+"/"
    foldernameInterpolation = foldernameTimeHorizon+"interpolation/"
    foldernameSignalDetector = foldernameTimeHorizon+"signal_detector/"
    foldernamePredictions = foldernameTimeHorizon+"predictions/"
    foldernameImages = foldernameTimeHorizon+"images/"
    foldernameImagesInterpolation = foldernameImages+"interpolation/"
    foldernameImagesSignalDetector = foldernameImages+"signal_detector/"


    ####### initialise interpolators #######
    interpolators[t] = attmoInterpolator(config, t, columnNamesInterpolator, foldernameInterpolation, foldernameImagesInterpolation)
    dcosInterpolation[t] = [[] for _ in range(len(config.thresholdsForInterpolation))]
    for i in range(len(config.thresholdsForInterpolation)):
        dcosInterpolation[t][i] = DcOS_TrendGenerator.DcOS(config.thresholdsForInterpolation[i], -1)


    ####### initialise signal detectors #######
    signalDetectors[t] = attmoSignalDetector(config, t, columnNamesSignalDetector, foldernameSignalDetector, foldernameImagesSignalDetector)
    eventsSignalDetector[t] = intrinsicTimeEventsSignalDetector(config.timeHorizons[t], config.thresholdsForInterpolation[t+9:t+12])
    dcosSignalDetection[t] = [[] for _ in range(3)]
    for j in range(3):
        dcosSignalDetection[t][j] = DcOS_TrendGenerator.DcOS(config.thresholdsForInterpolation[t+9+j], -1) #interpolators[t].interpolatedThresholds[j]
    crossSignals[t] = crossSignal()
    trendLineSignals[t] = [[] for _ in range(2)]
    trendLineSignals[t][0] = trendLineSignal(-1, config.plotData, foldernameImagesSignalDetector)
    trendLineSignals[t][1] = trendLineSignal(1, config.plotData, foldernameImagesSignalDetector)
    predictionGenerators[t] = predictionGenerator(config.timeHorizons[t], columnNamesPredictions, foldernamePredictions, config.savePredictionData, config.verbose)


def on_open(ws):
    print("Connection is open.")


def on_close(ws, status, message):
    print("Connection is closed.")


####### main loop #######
def on_message(ws, message):
    global closePrice #, midprice, price_1, df, client, assetString
    global config, tickReader, interpolators, dcosInterpolation
    global signalDetectors, dcosSignalDetection, eventsSignalDetector
    global crossSignals, trendLineSignals, predictionGenerators


    ####### 1. Read tick & update counters #######
    message = json.loads(message)
    df, midprice = functions.manipulation(message)
    tickReader = tickReader.run(df.timestamp[0], midprice)
    closePrice = DcOS_TrendGenerator.Price(tickReader.midprice, tickReader.midprice, tickReader.iteration)


    for t in range(len(config.timeHorizons)):

        ####### 2. run interpolation #######
        interpolators[t] = interpolators[t].run(dcosInterpolation[t], closePrice)
        #print(f"iteration block {interpolators[t].iterationBlock} / {config.blockLengths[t]}")

        ####### 3. run signal detection #######
        signalDetectors[t] = signalDetectors[t].update(config, tickReader, dcosSignalDetection[t], eventsSignalDetector[t], predictionGenerators[t], crossSignals[t], trendLineSignals[t], closePrice, interpolators[t].iterationBlock, interpolators[t].block)


        ####### 4. compute volatility #######
        if interpolators[t].iterationBlock == interpolators[t].blockLength:
            interpolators[t] = interpolators[t].interpolate(tickReader)
            eventsSignalDetector[t] = eventsSignalDetector[t].reset(dcosSignalDetection[t], closePrice, interpolators[t].interpolatedThresholds)


####### Run Program #######
def connect_to_binance_stream():
    socket = "wss://stream.binance.com:9443/stream?streams="+assetString
    ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
    ws.run_forever()


####### Initial connection #######
connect_to_binance_stream()
