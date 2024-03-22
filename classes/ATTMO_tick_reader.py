import pandas as pd


class ATTMO_tick_reader:
    __slots__ = ['iteration', 'timestamp', 'timestampUnderscore',
         'midprice', 'symbol_1', 'symbol_2', 'startTimeString']
    def __init__(self, config):
        self.iteration = 0
        #self.iterationBlock = 0
        #self.block = 0
        self.timestamp = ''
        self.midprice = 0
        self.symbol_1 = config.symbol_1
        self.symbol_2 = config.symbol_2
        self.startTimeString = ''
    def run(self, timest, mid, saveData, colNames_tick, foldername_ticks):
        self.iteration += 1
        #self.iterationBlock += 1
        self.midprice = mid
        tim = str(timest)
        tims = tim[:-7]
        timst = tims.replace(" ", "_")
        self.timestamp = timst.replace(":", "_")
        if self.iteration == 1:
            self.startTimeString = self.timestamp
            print(f"{self.startTimeString}: 1st tick received.")
        if saveData:
            df_tick = pd.DataFrame(columns=colNames_tick)
            df_tick.loc[0] = [self.iteration, self.timestamp, self.midprice]
            df_tick.to_parquet(f"{foldername_ticks}{self.timestamp}.parquet")
        return self
    #def reset(self):
    #    self.iterationBlock = 0
    #    self.block += 1
        return self
