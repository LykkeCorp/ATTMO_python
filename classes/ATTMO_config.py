import numpy as np


class ATTMO_config:
    __slots__ = ['symbol_1', 'symbol_2',
        'attmoForecastLabels', 'timeHorizons',
        'startingValuesTrendStrength', 'trendForecastLabels',
        'thresholdsForInterpolation', 'desiredEventFrequenciesList', 'blockLengths',
        'predictionFactor',
        'saveTickData', 'saveInterpolationDataFull', 'saveInterpolationDataLight', 'saveSignalDetectionData', 'saveInterpolationData',
        'runOnLocal', 'runOnNotebook', 'verbose', 'clearOutput']
    def __init__(self):
        with open('ATTMO_config.txt', 'r') as file:
            config_content = file.read()
        config_dict = {}
        exec(config_content, config_dict)

        # Asset pair (just 1 for the moment)
        self.symbol_1 = config_dict.get('symbol_1', None)
        self.symbol_2 = config_dict.get('symbol_2', None)

        # ATTMO output labels
        #self.attmoForecast = ['Stormy', 'Rainy', 'Cloudy', 'Sunny with some clouds', 'Sunny', 'Tropical']
        #self.timeHorizons = ['shortterm', '1h', '4h', '1d', '3d', '1w']
        #self.startingValuesTrendStrength = [9, 20, 42, 59, 75, 92]                                                                  # old ATTMO
        #self.trendForecast = ['bearish_very_extended', 'bearish_extended', 'bearish_very', 'bearish', 'bearish_neutral',            # old ATTMO
        #                        'bullish_neutral', 'bullish', 'bullish_very', 'bullish_extended', 'bullish_very_extended']
        self.attmoForecastLabels = config_dict.get('attmoForecastLabels', None)
        self.timeHorizons = config_dict.get('timeHorizons', None)
        self.startingValuesTrendStrength = config_dict.get('startingValuesTrendStrength', None)
        self.trendForecastLabels = config_dict.get('trendForecastLabels', None)

        # Interpolation settings
        #log_spaced_values = np.logspace(1, -3, 30)
        #self.thresholdsForInterpolation = list(np.round(np.flip(log_spaced_values)/100,5))
        self.thresholdsForInterpolation = config_dict.get('thresholdsForInterpolation', None)
        self.desiredEventFrequenciesList = config_dict.get('desiredEventFrequenciesList', None)
        #self.blockLength = self.desiredEventFrequenciesList[2]*100000
        self.blockLengths = config_dict.get('blockLengths', None)

        # Prediction settings
        self.predictionFactor = config_dict.get('predictionFactor', None)

        # Saving settings
        self.saveTickData = config_dict.get('saveTickData', None)
        self.saveInterpolationData = config_dict.get('saveInterpolationData', None)
        self.saveSignalDetectionData = config_dict.get('saveSignalDetectionData', None)
        savePredictionData = config_dict.get('saveSignalDetectionData', None)

        # Configuration settings
        self.runOnLocal = config_dict.get('runOnLocal', None)
        self.runOnNotebook = config_dict.get('runOnNotebook', None)
        self.verbose = config_dict.get('verbose', None)
        self.clearOutput = config_dict.get('clearOutput', None)
