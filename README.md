# Python code for new ATTMO implementation

Saves tick data for 1 venue and 1 asset pair.

Tick data are sent to 6 different interpolator objects, corresponding to the 5 ATTMO time horizons (1h, 4h, 1d, 3d, 1w) + a shorter time frame useful for debug (shortterm).

Each **interpolator** runs 30 exponentially-spaced DcOS objects capturing price changes of different sizes, from 0.001% to 10%. These thresholds are fixed.
At the end of a given number of ticks received (a 'block', whose size depend on the give time frame), a power-law function is fitted to the number of events observed.
The inverse of the power-law is then used to estimate current volatility, and determines thereby the DcOS thresholds of the signalGenerators (which runs 3 DcOS objects with ever-changing thresholds), such that events are produced at pre-determined frequencies (e.g., 2.5%, 1%, 0.25% for the 1-hour time horizon, and 0.125%, 0.05%, 0.0125% for the 1-week time horizon).

Each **signalGenerator** then uses the interactions between the events produced by its DcOS objects to generate putative prediction signals by implementing an event-based equivalent of the golden-cross/death-cross method. In short, it spits out a value from -3 to 3 capturing the confidence of the putative prediction, where more extreme values indicate a stronger signal and 0 indicates no signal. Whenever a putative prediction of a higher absolute value with respect to the ongoing one is generated, prediction is updated (level 3 and -3 'premium' predictions are never updated). These values are also mapped to a 1-100 *trendStrength* value and a *trendForecast* string (e.g., 'bullish_very', 'bearish_extended') for backwards compatibility.
Confidence level on time: 0 ~ 7%, 1 ~ 38%, 2 ~ 33%, 3 ~ 22% (tested up to the 4h time horizon, but self-similarity should apply as results are consistent over the 3 shorter time horizons).
If the confidence level is 3 or -3, the signal is passed to the corresponding predictor object.

Each **predictor object** then uses the scaling-law parameters to place a target and a stop-loss that are consistent with current volatility (i.e., further apart if volatility is high and vice-versa). It also spits out the weather forecast (e.g., 'sunny', 'cloudy').
Accuracy of the predictions: shortterm (~25 min) ~ 61%, 1h ~ 58%, 4h ~ 55%. More data are needed to assess longer time horizons.
