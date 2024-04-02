import pandas as pd


class attmoTickReader:
    __slots__ = ['iteration', 'timestamp', 'timestampUnderscore',
         'midprice', 'symbol_1', 'symbol_2', 'startTimeString',
         'saveData', 'colNamesDf', 'outputDir']
    def __init__(self, config, colNames_tick, foldername_ticks):
        self.iteration = 0
        self.timestamp = ''
        self.midprice = 0
        self.symbol_1 = config.symbol_1
        self.symbol_2 = config.symbol_2
        self.startTimeString = ''
        self.saveData = config.saveTickData
        self.colNamesDf = colNames_tick
        self.outputDir = foldername_ticks
    def run(self, timest, mid):
        self.iteration += 1
        self.midprice = mid
        tim = str(timest)
        tims = tim[:-7]
        timst = tims.replace(" ", "_")
        self.timestamp = timst.replace(":", "_")
        if self.iteration == 1:
            self.startTimeString = self.timestamp
            print(f"{self.startTimeString}: 1st tick received.")
        if self.saveData:
            df_tick = pd.DataFrame(columns=self.colNamesDf)
            df_tick.loc[0] = [self.iteration, self.timestamp, self.midprice]
            df_tick.to_parquet(f"{self.outputDir}{self.timestamp}.parquet")
        return self
