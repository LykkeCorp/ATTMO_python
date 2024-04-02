import numpy as np


class attmoConfig:
    __slots__ = ['symbol_1', 'symbol_2',
        'attmoForecastLabels', 'timeHorizons',
        'startingValuesTrendStrength', 'trendForecastLabels',
        'thresholdsForInterpolation', 'desiredEventFrequenciesList', 'blockLengths',
        'predictionFactor',
        'saveTickData', 'saveSignalDetectionData', 'saveInterpolationData', 'savePredictionData',
        'runOnLocal', 'verbose']
    def __init__(self):
        with open('ATTMO_config.txt', 'r') as file:
            config_content = file.read()
        config_dict = {}
        exec(config_content, config_dict)

        # Asset pair (just 1 for the moment)
        self.symbol_1 = config_dict.get('symbol_1', None)
        self.symbol_2 = config_dict.get('symbol_2', None)

        # ATTMO output labels
        self.attmoForecastLabels = config_dict.get('attmoForecastLabels', None)
        self.timeHorizons = config_dict.get('timeHorizons', None)
        self.startingValuesTrendStrength = config_dict.get('startingValuesTrendStrength', None)
        self.trendForecastLabels = config_dict.get('trendForecastLabels', None)

        # Interpolation settings
        self.thresholdsForInterpolation = config_dict.get('thresholdsForInterpolation', None)
        self.desiredEventFrequenciesList = config_dict.get('desiredEventFrequenciesList', None)
        self.blockLengths = config_dict.get('blockLengths', None)

        # Prediction settings
        self.predictionFactor = config_dict.get('predictionFactor', None)

        # Saving settings
        self.saveTickData = config_dict.get('saveTickData', None)
        self.saveInterpolationData = config_dict.get('saveInterpolationData', None)
        self.saveSignalDetectionData = config_dict.get('saveSignalDetectionData', None)
        self.savePredictionData = config_dict.get('saveSignalDetectionData', None)

        # Configuration settings
        self.runOnLocal = config_dict.get('runOnLocal', None)
        self.verbose = config_dict.get('verbose', None)
