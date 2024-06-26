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
        #foldername_predictions_generated = foldername_time_horizon+"predictions_generated/"
        #foldername_predictions_outcome = foldername_time_horizon+"predictions_outcome/"
        foldername_images = foldername_time_horizon+"images/"
        os.makedirs(foldername_interpolation)
        os.makedirs(foldername_signal_detector)
        os.makedirs(foldername_predictions)
        #os.makedirs(foldername_predictions_generated)
        #os.makedirs(foldername_predictions_outcome)
        os.makedirs(foldername_images)
        foldername_images_interpolation = foldername_images + "interpolation/"
        os.makedirs(foldername_images_interpolation)
        foldername_images_signal_detector = foldername_images + "signal_detector/"
        os.makedirs(foldername_images_signal_detector)


def generate_dataframes_column_names():
    # tickReader
    columnNamesTickReader = ["iteration", "timestamp", "midprice"]

    # interpolator
    columnNamesInterpolator = ["iteration", "timestamp", "midprice", "iterationBlock", "block", "delta0", "delta1", "delta2", "paramA", "paramB", "rSquared", "windLevel", "windLabel"]

    # signalDetector
    columnNamesSignalDetector = ["iteration", "timestamp", "midprice", "iterationBlock", "block"]
    colNames_sd = [(f"threshold{j}", f"currentEvent{j}", f"nrOfEventsInBlock{j}") for j in range(3)]
    colNames_SD = [item for t in colNames_sd for item in t]
    columnNamesSignalDetector.extend(colNames_SD)
    columnNamesSignalDetector.extend(['signalDetected', 'currentForecastLevel', 'trendStrength', 'trendForecast', 'attmoForecast', # 'supportLineEstimationIteration',
                    'supportLineIntercept', 'supportLineSlope', 'supportLineRSquared', 'supportLineEstimationPoints',
                    'supportLineFirstSample', 'supportLineFirstTimestamp', 'supportLineFirstMidprice',
                    'supportLineLastSample', 'supportLineLastTimestamp', 'supportLineLastMidprice',
                    'resistanceLineIntercept', 'resistanceLineSlope', 'resistanceLineRSquared', 'resistanceLineEstimationPoints', # 'resistanceLineEstimationIteration',
                    'resistanceLineFirstSample', 'resistanceLineFirstTimestamp', 'resistanceLineFirstMidprice',
                    'resistanceLineLastSample', 'resistanceLineLastTimestamp', 'resistanceLineLastMidprice'])

    ## predictor
    #columnNamesPredictionGenerated = ["iteration", "timestamp", "midprice", "iterationBlock", "block",
    #        'predictionPriceChangePt', 'predictionDirection', 'attmoForecast', 'target', 'stopLoss',
    #        'predictionDurationTicks', 'predictionOutcome', 'nrTargetReached', 'nrStopLossReached']

    #columnNamesPredictionOutcome = ["iteration", "timestamp", "midprice", "iterationBlock", "block",
    #        'iterationPredictionStart', 'midpricePredictionStart', 'timestampPredictionStart',
    #        'attmoForecast', 'target', 'stopLoss',
    #        'predictionDurationTicks', 'predictionOutcome', 'nrTargetReached', 'nrStopLossReached']

    columnNamesPredictions =  ['iterationPredictionStart', 'timestampPredictionStart', 'midpricePredictionStart',
            'predictionPriceChangePt', 'predictionDirection', 'signal', 'target', 'stopLoss',
            'iterationPredictionEnd', 'timestampPredictionEnd', 'midpricePredictionEnd',
            'predictionDurationTicks', 'predictionOutcome', 'nrTargetReached', 'nrStopLossReached']

    return columnNamesTickReader, columnNamesInterpolator, columnNamesSignalDetector, columnNamesPredictions


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
