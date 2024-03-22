import glob
import os
from pathlib import Path
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd


#######################################################################################################
class DcOS:
    __slots__ = ['threshold', 'mode', 'extreme', 'prevExtreme', 'reference',
        'prevDC', 'DC', 'osL', 'totalMove', 'nOS', 'dcL']

    def __init__(self, threshold, initialMode):
        self.threshold = threshold
        self.mode = initialMode
        self.extreme = self.prevExtreme = self.reference = self.prevDC = self.DC = Event(0, 0)
        self.osL = 0
        self.totalMove = 0
        self.nOS = 0
        self.dcL = 0
        #self.NaT = []

    def run(self, aPrice):
        return self.runRelative(aPrice)

    # Uses log function in order to compute a price move
    # @param aPrice is a new price
    # @return +1 or -1 in case of Directional-Change (DC) Intrinsic Event (IE) upward or downward and also +2 or -2
    # in case of an Overshoot (OS) IE upward or downward. Otherwise, returns 0;
    def runRelative(self, aPrice):
        current_price = Event(aPrice.getMid(), aPrice.time)

        #if pd.isnull(current_price.level):
        #    #print("NaT!")
        #    self.nat.append(i)
        #else:
        if (self.extreme.level == 0):
            self.extreme = self.prevExtreme = self.reference = self.prevDC = self.DC = current_price
        side = -self.mode

        # code below is key: checks if current_price has moved in the direction of the current mode of the algorithm,
        # i.e., if the current price is increasing while in bullish mode or decreasing while in bearish mode.
        # If the condition is true, the algorithm continues to track the current trend by updating the extreme value.
        # Otherwise, it checks for the occurrence of a directional change or overshoot event,
        # which would cause the algorithm to switch modes and update the relevant variables accordingly.
        if (current_price.level * side > side * self.extreme.level): # If price moves in same direction...
            self.extreme = current_price
            # Examples:
            # - if ‘current_price.level’ = 3, ‘self.extreme.level’ = 2, and ‘side’ = 1, expression would evaluate to
            #     True, as the price is increasing while in bullish mode
            # - if ‘current_price.level’ = 3, ‘self.extreme.level’ = 2, and ‘side’ = -1, expression would evaluate to
            #     False, as the price is increasing while in bearish mode
            # - if ‘current_price.level’ = 2, ‘self.extreme.level’ = 3, and ‘side’ = -1, expression would evaluate to
            #     True, as the price is decreasing while in bearish mode

            # The lines of code below are checking if there has been an overshoot event (OS) and if so,
            # updating the reference price to the new extreme price and returning a dcos value of +2 or -2
            # depending on the direction of the overshoot.

            #If the absolute value of the logarithmic difference between the extreme price and the reference price
            # is greater than or equal to the threshold, then the reference price is updated with the extreme price
            # and the nOS variable is incremented.
            if (side * math.log(self.extreme.level / self.reference.level) >= self.threshold): # IF OVERSHOOT
                self.reference = self.extreme
                self.nOS = self.nOS + 1
                return 2 * side
            return 0

        else: # If price moves in opposite direction...
            self.dcL = - side * math.log(current_price.level / self.extreme.level)

            if (self.dcL >= self.threshold): # IF DIRECTIONAL CHANGE
                self.osL = side * math.log(self.extreme.level / self.DC.level)
                self.totalMove = side * math.log(self.extreme.level / self.prevExtreme.level)
                self.prevDC = self.DC
                self.DC = current_price
                self.prevExtreme = self.extreme
                self.extreme = self.reference = current_price
                self.mode *= -1
                self.nOS = 0
                return -side * 1
        return 0

    #######################################################################################################
class Price:
    __slots__ = ['bid', 'ask', 'time']
    def __init__(self, bid, ask, time):
        self.bid = bid
        self.ask = ask
        self.time = time
    def getMid(self):
        return (self.bid + self.ask) / 2

    #######################################################################################################
class Event:
    __slots__ = ['level', 'time']
    def __init__(self, level, time):
        self.level = level
        self.time = time
