import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


####### Functions #######
def create_folders(foldername, timeHorizons):
    foldername_ticks = foldername+"ticks/"
    os.makedirs(foldername_ticks)
    for t in range(len(timeHorizons)):
        foldername_time_horizon = foldername+timeHorizons[t]+"/"
        os.makedirs(foldername_time_horizon)
        foldername_interpolation = foldername_time_horizon+"interpolation/"
        foldername_signal_detector = foldername_time_horizon+"signal_detector/"
        foldername_predictions = foldername_time_horizon+"predictions/"
        os.makedirs(foldername_interpolation)
        os.makedirs(foldername_signal_detector)
        os.makedirs(foldername_predictions)


def generate_dataframes_column_names():
    # tickReader
    columnNamesTickReader = ["iteration", "timestamp", "midprice"]

    # interpolator
    columnNamesInterpolator = ["iteration", "timestamp", "midprice", "iterationBlock", "block", "delta_0", "delta_1", "delta_2", "param_a", "param_b", "r_squared"]

    # signalDetector
    columnNamesSignalDetector = ["iteration", "timestamp", "midprice", "iterationBlock", "block"]
    colNames_sd = [(f"threshold_{j}", f"currentEvent_{j}", f"nrOfEventsInBlock_{j}") for j in range(3)]
    colNames_SD = [item for t in colNames_sd for item in t]
    columnNamesSignalDetector.extend(colNames_SD)
    columnNamesSignalDetector.extend(['currentSignalLevel', 'trendStrength', 'trendForecast'])

    # predictor
    columnNamesPrediction = ["iteration", "timestamp", "midprice", "iterationBlock", "block",
            'iterationPredictionStart', 'timestampPredictionStart', 'midpricePredictionStart',
            'iterationPredictionEnd', 'midpricePredictionEnd',  'timestampPredictionEnd',
            'predictionPriceChangePt', 'predictionDirection', 'target', 'stopLoss',
            'attmoForecast', 'predictionDurationTicks', 'predictionOutcome', 'nrTargetReached', 'nrStopLossReached']
    return columnNamesTickReader, columnNamesInterpolator, columnNamesSignalDetector, columnNamesPrediction


def copy_configuration_file(foldername, config_file_str, now, symbol_1, symbol_2):
    with open(config_file_str, 'r') as source_file:
        config_content = source_file.read()
    with open(f"{foldername}config_{symbol_1}{symbol_2}_{now}.txt", 'w') as target_file:
        target_file.write(config_content)


def manipulation(source):
    rel_data = source['data']['k']['c']
    evt_time = pd.to_datetime(source['data']['E'], unit='ms')
    df = pd.DataFrame(rel_data, columns=[source['data']['s']],
                      index=[evt_time])
    df.index.name = 'timestamp'
    df = df.astype(float)
    df = df.reset_index()
    val_1 = df.BTCFDUSD.values[0] #### important!!!!!!!!!!!!!
    return df, val_1
