import numpy as np
import pandas as pd
import glob
import os

import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from classes.attmoConfig import attmoConfig
config = attmoConfig()


def find_first_non_zero_indices(lst):
    non_zero_indices = []
    series_started = False
    for i, num in enumerate(lst):
        if num != 0 and not series_started:
            non_zero_indices.append(i)
            series_started = True
        elif num == 0:
            series_started = False
    return non_zero_indices


def find_last_non_zero_indices(lst):
    non_zero_indices = []
    last_non_zero_index = None
    for i, num in enumerate(lst):
        if num != 0:
            last_non_zero_index = i
        elif num == 0 and last_non_zero_index is not None:
            non_zero_indices.append(last_non_zero_index)
            last_non_zero_index = None
    # Append the index of the last non-zero element if the list ends with non-zero values
    if last_non_zero_index is not None:
        non_zero_indices.append(last_non_zero_index)
    return non_zero_indices


def runPostprocess(folderpath, symbol_1, symbol_2, date, plotSplitted, plotSingleTrace, xTickSpacing):
    foldername = f"{folderpath}{symbol_1}{symbol_2}_{date}/"

    # load saved cfg file
    with open(f"{foldername}config_{symbol_1}{symbol_2}_{date}.txt", 'r') as source_file:
        config_content = source_file.read()
        config_dict = {}
        exec(config_content, config_dict)

        # Asset pair (just 1 for the moment)
        timeHorizons = config_dict.get('timeHorizons', None)
        blockLengths = config_dict.get('blockLengths', None)


    results_DF = pd.DataFrame(columns = ['timeHorizon', 'forecast_duration_X', 'forecast_duration_SD', 'pred_duration_X', 'pred_duration_SD',
                                        'overall_n_pred', 'n_lvl_1_pred', 'n_lvl_2_pred', 'n_lvl_3_pred',
                                        'overall_accuracy', 'pred_accuracy_lvl_1', 'pred_accuracy_lvl_2', 'pred_accuracy_lvl_3'])

    for t in range(len(timeHorizons)):
        chunk_size = blockLengths[t]*10

        foldername_time_horizon = foldername+timeHorizons[t]+"/"
        foldername_interpolation = foldername_time_horizon+"interpolation/"
        foldername_signal_detector = foldername_time_horizon+"signal_detector/"
        foldername_predictions = foldername_time_horizon+"predictions/"

        file_path = foldername_time_horizon + f"DF_signals_{timeHorizons[t]}.csv"
        if os.path.exists(file_path):
            print(f"Time horizon = {timeHorizons[t]}. Signal detector file exists. Loading...")
            DF = pd.read_csv(foldername_time_horizon + f"DF_signals_{timeHorizons[t]}.csv")
        else:
            print(f"Time horizon = {timeHorizons[t]}. Creating signal detector file...")
            event_files = glob.glob(f"{foldername_signal_detector}*.parquet")
            for i in range(len(event_files)):
                if i == 0:
                    DF = pd.read_parquet(event_files[i])
                elif i > 0:
                    df = pd.read_parquet(event_files[i])
                    DF = pd.concat([DF, df])
            DF.to_csv(foldername_time_horizon + f"DF_signals_{timeHorizons[t]}.csv")

        ie_signal = DF[abs(DF.signalDetected) > 0]
        ie_signal_1 = DF[abs(DF.signalDetected) == 1]
        ie_signal_2 = DF[abs(DF.signalDetected) == 2]
        ie_signal_3 = DF[abs(DF.signalDetected) == 3]

        ### ATTMO forecast
        a = ie_signal.currentForecastLevel.values
        indices = []
        for i in range(1, len(ie_signal)):
            if a[i] != a[i - 1]:
                indices.append(i)

        ie_forecast = ie_signal.copy()
        ie_forecast = ie_forecast.iloc[indices]



        ### load interpolation files
        file_path = foldername_time_horizon + f"DF_interpolation_{timeHorizons[t]}.csv"
        if os.path.exists(file_path):
            print(f"Time horizon = {timeHorizons[t]}. Interpolation file exists. Loading...")
            DF_interp = pd.read_csv(foldername_time_horizon + f"DF_interpolation_{timeHorizons[t]}.csv")
        else:
            print(f"Time horizon = {timeHorizons[t]}. Creating interpolation file...")
            event_files = glob.glob(f"{foldername_interpolation}*.parquet")
            for i in range(len(event_files)):
                if i == 0:
                    DF_interp = pd.read_parquet(event_files[i])
                elif i > 0:
                    df_interp = pd.read_parquet(event_files[i])
                    DF_interp = pd.concat([DF_interp, df_interp])
            DF_interp.to_csv(foldername_time_horizon + f"DF_interpolation_{timeHorizons[t]}.csv")



        ### load prediction files
        foldername_predictions = foldername_time_horizon+"predictions/"

        file_path = foldername_time_horizon + f"DF_predictions_{timeHorizons[t]}.csv"
        if os.path.exists(file_path):
            print(f"Time horizon = {timeHorizons[t]}. Predictions file exists. Loading...")
            DF_pred = pd.read_csv(foldername_time_horizon + f"DF_predictions_{timeHorizons[t]}.csv")
        else:
            print(f"Time horizon = {timeHorizons[t]}. Creating predictions file...")
            event_files = glob.glob(f"{foldername_predictions}*.parquet")
            for i in range(len(event_files)):
                if i == 0:
                    DF_pred = pd.read_parquet(event_files[i])
                else:
                    df_pred = pd.read_parquet(event_files[i])
                    DF_pred = pd.concat([DF_pred, df_pred])
            DF_pred.to_csv(foldername_time_horizon + f"DF_predictions_{timeHorizons[t]}.csv")

        ### accuracy trace
        ie_pred_correct = DF_pred.loc[DF_pred.predictionOutcome == 1]
        ie_pred_incorrect = DF_pred.loc[DF_pred.predictionOutcome == -1]


        ### signal level trace
        ie_pred_lvl_1 = DF_pred.loc[abs(DF_pred.signal) == 1]
        ie_pred_lvl_2 = DF_pred.loc[abs(DF_pred.signal) == 2]
        ie_pred_lvl_3 = DF_pred.loc[abs(DF_pred.signal) == 3]


        ### accuracy * level trace
        ie_pred_correct_lvl_1 = ie_pred_lvl_1.loc[ie_pred_lvl_1.predictionOutcome == 1]
        ie_pred_incorrect_lvl_1 = ie_pred_lvl_1.loc[ie_pred_lvl_1.predictionOutcome == -1]
        ie_pred_correct_lvl_2 = ie_pred_lvl_2.loc[ie_pred_lvl_2.predictionOutcome == 1]
        ie_pred_incorrect_lvl_2 = ie_pred_lvl_2.loc[ie_pred_lvl_2.predictionOutcome == -1]
        ie_pred_correct_lvl_3 = ie_pred_lvl_3.loc[ie_pred_lvl_3.predictionOutcome == 1]
        ie_pred_incorrect_lvl_3 = ie_pred_lvl_3.loc[ie_pred_lvl_3.predictionOutcome == -1]


        ### descriptives
        forecast_duration = [0]
        for i in range(len(ie_forecast)):
            forecast_duration.append(ie_forecast.iteration.iloc[i] - forecast_duration[i])
        forecast_durations = forecast_duration[1:]
        forecast_duration_X = np.round(np.mean(forecast_durations),2)
        forecast_duration_SD = np.round(np.std(forecast_durations),2)

        pred_duration_X = np.round(np.mean(np.array(DF_pred.iterationPredictionEnd)-np.array(DF_pred.iterationPredictionStart)))
        pred_duration_SD = np.round(np.mean(np.array(DF_pred.iterationPredictionEnd)-np.array(DF_pred.iterationPredictionStart)))

        overall_n_pred = len(ie_pred_correct)+len(ie_pred_incorrect)
        n_lvl_1_pred = len(ie_pred_correct_lvl_1)+len(ie_pred_incorrect_lvl_1)
        n_lvl_2_pred = len(ie_pred_correct_lvl_2)+len(ie_pred_incorrect_lvl_2)
        n_lvl_3_pred = len(ie_pred_correct_lvl_3)+len(ie_pred_incorrect_lvl_3)

        overall_accuracy = np.round(len(ie_pred_correct) * 100 / (len(ie_pred_correct)+len(ie_pred_incorrect)),2)

        pred_accuracy_lvl_1 = 0
        pred_accuracy_lvl_2 = 0
        pred_accuracy_lvl_3 = 0

        if n_lvl_1_pred > 0:
            pred_accuracy_lvl_1 = np.round(len(ie_pred_correct_lvl_1) * 100 / (len(ie_pred_correct_lvl_1)+len(ie_pred_incorrect_lvl_1)),2)
        if n_lvl_2_pred > 0:
            pred_accuracy_lvl_2 = np.round(len(ie_pred_correct_lvl_2) * 100 / (len(ie_pred_correct_lvl_2)+len(ie_pred_incorrect_lvl_2)),2)
        if n_lvl_3_pred > 0:
            pred_accuracy_lvl_3 = np.round(len(ie_pred_correct_lvl_3) * 100 / (len(ie_pred_correct_lvl_3)+len(ie_pred_incorrect_lvl_3)),2)


        print("")
        print(f"{timeHorizons[t]}:")
        print(f"Mean forecast duration = {np.round(forecast_duration_X/60)} min. (SD = {np.round(forecast_duration_SD/60)}).")
        print(f"Number of blocks = {len(DF_interp)}.")
        print(f"Mean prediction duration = {np.round(pred_duration_X/60)} min. (SD = {np.round(pred_duration_SD/60)}).")
        print(f"Tot predictions generated = {overall_n_pred}: {n_lvl_1_pred} (lvl. 1), {n_lvl_2_pred} (lvl. 2), and {n_lvl_3_pred} (lvl. 3).")
        print(f"Overall accuracy = {overall_accuracy} %.")
        print(f"Accuracy lvl. 1 = {pred_accuracy_lvl_1} %.")
        print(f"Accuracy lvl. 2 = {pred_accuracy_lvl_2} %.")
        print(f"Accuracy lvl. 3 = {pred_accuracy_lvl_3} %.")

        results_DF.loc[t] = [timeHorizons[t], forecast_duration_X, forecast_duration_SD,
                            pred_duration_X, pred_duration_SD,
                            overall_n_pred, n_lvl_1_pred, n_lvl_2_pred, n_lvl_3_pred,
                            overall_accuracy, pred_accuracy_lvl_1, pred_accuracy_lvl_2, pred_accuracy_lvl_3]

        if plotSplitted:
            plotSplittedTrace(DF, DF_pred, DF_interp, chunk_size, timeHorizons[t], foldername_time_horizon, xTickSpacing)
        if plotSingleTrace:
            plotInOneImage(DF, DF_pred, DF_interp, timeHorizons[t], foldername_time_horizon, xTickSpacing)

    results_DF.to_csv(foldername + f"descriptives.csv")
    return results_DF


def plotSplittedTrace(DF, DF_pred, DF_interp, chunk_size, timeHorizon, foldername_time_horizon, xTickSpacing):
    num_chunks = int(len(DF) // chunk_size + (1 if len(DF) % chunk_size != 0 else 0))

    for chunk_index in range(num_chunks):
        start_index = chunk_index * chunk_size
        end_index = min((chunk_index + 1) * chunk_size, len(DF))

        chunk_df = DF.iloc[start_index:end_index]



        ### create figure
        fn = f"attmo_forecast_{timeHorizon}"
        file_name = foldername+fn+".html"
        init_notebook_mode(connected=True)
        col_seq = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

        yaxis = dict(
            title=f'test_image',
            showgrid=True,
            gridcolor='white',
            zeroline=False
        )

        fig = go.Figure()



        ### dcos traces
        ie_A = chunk_df[abs(chunk_df.currentEvent0) > 0]
        ie_A_dc = chunk_df[abs(chunk_df.currentEvent0) == 1]
        ie_A_os = chunk_df[abs(chunk_df.currentEvent0) == 2]
        ie_B = chunk_df[abs(chunk_df.currentEvent1) > 0]
        ie_B_dc = chunk_df[abs(chunk_df.currentEvent1) == 1]
        ie_B_os = chunk_df[abs(chunk_df.currentEvent1) == 2]
        ie_C = chunk_df[abs(chunk_df.currentEvent2) > 0]
        ie_C_dc = chunk_df[abs(chunk_df.currentEvent2) == 1]
        ie_C_os = chunk_df[abs(chunk_df.currentEvent2) == 2]

        trace_mid = go.Scatter(
            x=chunk_df.iteration,
            y=chunk_df.midprice,
            yaxis='y',
            name="midprice",
            line=dict(color='gray', width=0.5)
        )


        traceA = go.Scatter(
            x=ie_A.iteration,
            y=ie_A.midprice,
            yaxis='y',
            name="delta interp. A",
            line=dict(color=col_seq[0], width=0.5)
        )
        traceA_dc = go.Scatter(
            x=ie_A_dc.iteration,
            y=ie_A_dc.midprice,
            name='delta interp. A DC',
            yaxis='y',
            mode='markers',
            marker=dict(
                    color='white',
                    symbol='square',
                    size=2,
                    line=dict(
                        color=col_seq[0],
                        width=0.5
                    )
                )
        )
        traceA_os = go.Scatter(
            x=ie_A_os.iteration,
            y=ie_A_os.midprice,
            name='delta interp. A OS',
            yaxis='y',
            mode='markers',
            marker=dict(
                      color='black',
                      symbol='circle',
                      size=2,
                      line=dict(
                          color=col_seq[0],
                          width=0.5
                      )
                  )
        )

        traceB = go.Scatter(
            x=ie_B.iteration,
            y=ie_B.midprice,
            yaxis='y',
            name="delta interp. B",
            line=dict(color=col_seq[1], width=1)
        )
        traceB_dc = go.Scatter(
            x=ie_B_dc.iteration,
            y=ie_B_dc.midprice,
            name='delta interp. B DC',
            yaxis='y',
            mode='markers',
            marker=dict(
                    color='white',
                    symbol='square',
                    size=3,
                    line=dict(
                        color=col_seq[1],
                        width=1
                    )
                )
        )
        traceB_os = go.Scatter(
            x=ie_B_os.iteration,
            y=ie_B_os.midprice,
            name='delta interp. B OS',
            yaxis='y',
            mode='markers',
            marker=dict(
                      color='black',
                      symbol='circle',
                      size=3,
                      line=dict(
                          color=col_seq[1],
                          width=1
                      )
                  )
        )

        traceC = go.Scatter(
            x=ie_C.iteration,
            y=ie_C.midprice,
            yaxis='y',
            name="delta interp. C",
            line=dict(color=col_seq[2], width=1.5)
        )
        traceC_dc = go.Scatter(
            x=ie_C_dc.iteration,
            y=ie_C_dc.midprice,
            name='delta interp. C DC',
            yaxis='y',
            mode='markers',
            marker=dict(
                    color='white',
                    symbol='square',
                    size=4,
                    line=dict(
                        color=col_seq[2],
                        width=1.5
                    )
                )
        )
        traceC_os = go.Scatter(
            x=ie_C_os.iteration,
            y=ie_C_os.midprice,
            name='delta interp. C OS',
            yaxis='y',
            mode='markers',
            marker=dict(
                      color='black',
                      symbol='circle',
                      size=4,
                      line=dict(
                          color=col_seq[2],
                          width=1.5
                      )
                  )
        )

        fig.add_trace(trace_mid)
        fig.add_trace(traceA)
        fig.add_trace(traceA_dc)
        fig.add_trace(traceA_os)
        fig.add_trace(traceB)
        fig.add_trace(traceB_dc)
        fig.add_trace(traceB_os)
        fig.add_trace(traceC)
        fig.add_trace(traceC_dc)
        fig.add_trace(traceC_os)



        ### trend lines
        idxStartRes = find_first_non_zero_indices(chunk_df.resistanceLineFirstSample)
        idxLastRes = find_last_non_zero_indices(chunk_df.resistanceLineLastSample)
        idxStartSup = find_first_non_zero_indices(chunk_df.supportLineFirstSample)
        idxLastSup = find_last_non_zero_indices(chunk_df.supportLineLastSample)

        ie_res_line_start = chunk_df.iloc[idxStartRes]
        ie_sup_line_start = chunk_df.iloc[idxStartSup]
        ie_res_line_end = chunk_df.iloc[idxLastRes]
        ie_sup_line_end = chunk_df.iloc[idxLastSup]

        for i in range(len(ie_sup_line_end)):
            x_values = [ie_sup_line_end.supportLineFirstSample.iloc[i], ie_sup_line_end.supportLineLastSample.iloc[i]]
            y_values = [ie_sup_line_end.supportLineFirstMidprice.iloc[i], ie_sup_line_end.supportLineLastMidprice.iloc[i]]
            trace_sup_line = go.Scatter(x=x_values,
                                        y=y_values,
                                        yaxis='y',
                                        showlegend=False,
                                        line=dict(color='green', width=3))
            fig.add_trace(trace_sup_line)

        for i in range(len(ie_res_line_end)):
            x_values = [ie_res_line_end.resistanceLineFirstSample.iloc[i], ie_res_line_end.resistanceLineLastSample.iloc[i]]
            y_values = [ie_res_line_end.resistanceLineFirstMidprice.iloc[i], ie_res_line_end.resistanceLineLastMidprice.iloc[i]]
            trace_res_line = go.Scatter(x=x_values,
                                        y=y_values,
                                        yaxis='y',
                                        showlegend=False,
                                        line=dict(color='red', width=3))
            fig.add_trace(trace_res_line)



        ### forecast
        ie_signal = chunk_df[abs(chunk_df.signalDetected) > 0]
        ie_signal_1 = chunk_df[abs(chunk_df.signalDetected) == 1]
        ie_signal_2 = chunk_df[abs(chunk_df.signalDetected) == 2]
        ie_signal_3 = chunk_df[abs(chunk_df.signalDetected) == 3]

        a = ie_signal.currentForecastLevel.values
        indices = []
        for i in range(1, len(ie_signal)):
            if a[i] != a[i - 1]:
                indices.append(i)

        ie_forecast = ie_signal.copy()
        ie_forecast = ie_forecast.iloc[indices]

        for i in range(len(ie_forecast)-1):
            if ie_forecast.currentForecastLevel.iloc[i] == -3:
                fillcol = '#87CEFA'
            elif ie_forecast.currentForecastLevel.iloc[i] == -2:
                fillcol = '#4169E1'
            elif ie_forecast.currentForecastLevel.iloc[i] == -1:
                fillcol = '#191970'
            elif ie_forecast.currentForecastLevel.iloc[i] == 0:
                fillcol = '#F0F8FF'
            elif ie_forecast.currentForecastLevel.iloc[i] == 1:
                fillcol = '#FFFF66'
            elif ie_forecast.currentForecastLevel.iloc[i] == 1:
                fillcol = '#FFD700'
            elif ie_forecast.currentForecastLevel.iloc[i] == 3:
                fillcol = '#FFA500'

            fig.add_vrect(x0=ie_forecast.iteration.iloc[i], x1=ie_forecast.iteration.iloc[i+1],
                          annotation_text=str(ie_forecast.attmoForecast.iloc[i]), annotation_position="top left",
                          fillcolor=fillcol, opacity=0.25, line_width=0)



        ### prediction traces
        idx = list(np.where((DF_pred.iterationPredictionStart>start_index) & (DF_pred.iterationPredictionEnd<end_index))[0])

        if len(idx) > 0:
            chunk_df_pred = DF_pred.iloc[idx[0]:idx[len(idx)-1]]


            ### accuracy trace
            ie_pred_correct = chunk_df_pred.loc[chunk_df_pred.predictionOutcome == 1]
            ie_pred_incorrect = chunk_df_pred.loc[chunk_df_pred.predictionOutcome == -1]


            ### signal level * accuracy trace
            ie_pred_correct_lvl_1 = ie_pred_correct.loc[ie_pred_correct.signal == 1]
            ie_pred_incorrect_lvl_1 = ie_pred_incorrect.loc[ie_pred_incorrect.signal == -1]
            ie_pred_correct_lvl_2 = ie_pred_correct.loc[ie_pred_correct.signal == 2]
            ie_pred_incorrect_lvl_2 = ie_pred_incorrect.loc[ie_pred_incorrect.signal == -2]
            ie_pred_correct_lvl_3 = ie_pred_correct.loc[ie_pred_correct.signal == 3]
            ie_pred_incorrect_lvl_3 = ie_pred_incorrect.loc[ie_pred_incorrect.signal == -3]


            trace_pred_correct_lvl_1 = go.Scatter(
                x=ie_pred_correct_lvl_1.iterationPredictionStart,
                y=ie_pred_correct_lvl_1.midpricePredictionStart,
                yaxis='y',
                name=f"accurate lvl. 1 predictions",
                mode='markers',
                marker=dict(
                        color='#00FF00',
                        symbol='diamond',
                        size=5,
                        line=dict(
                            color='#006400',
                            width=3
                        )
                    )
            )
            trace_pred_incorrect_lvl_1 = go.Scatter(
                x=ie_pred_incorrect_lvl_1.iterationPredictionStart,
                y=ie_pred_incorrect_lvl_1.midpricePredictionStart,
                yaxis='y',
                name=f"inaccurate lvl. 1 predictions",
                mode='markers',
                marker=dict(
                        color='#FF0000',
                        symbol='diamond',
                        size=5,
                        line=dict(
                            color='#DC143C',
                            width=3
                        )
                    )
            )

            trace_pred_correct_lvl_2 = go.Scatter(
                x=ie_pred_correct_lvl_2.iterationPredictionStart,
                y=ie_pred_correct_lvl_2.midpricePredictionStart,
                yaxis='y',
                name=f"accurate lvl. 2 predictions",
                mode='markers',
                marker=dict(
                        color='#00FF00',
                        symbol='diamond',
                        size=5,
                        line=dict(
                            color='#006400',
                            width=2
                        )
                    )
            )
            trace_pred_incorrect_lvl_2 = go.Scatter(
                x=ie_pred_incorrect_lvl_2.iterationPredictionStart,
                y=ie_pred_incorrect_lvl_2.midpricePredictionStart,
                yaxis='y',
                name=f"inaccurate lvl. 2 predictions",
                mode='markers',
                marker=dict(
                        color='#FF0000',
                        symbol='diamond',
                        size=5,
                        line=dict(
                            color='#DC143C',
                            width=2
                        )
                    )
            )

            trace_pred_correct_lvl_3 = go.Scatter(
                x=ie_pred_correct_lvl_3.iterationPredictionStart,
                y=ie_pred_correct_lvl_3.midpricePredictionStart,
                yaxis='y',
                name=f"accurate lvl. 3 predictions",
                mode='markers',
                marker=dict(
                        color='#00FF00',
                        symbol='diamond',
                        size=5,
                        line=dict(
                            color='#006400',
                            width=3
                        )
                    )
            )
            trace_pred_incorrect_lvl_3 = go.Scatter(
                x=ie_pred_incorrect_lvl_3.iterationPredictionStart,
                y=ie_pred_incorrect_lvl_3.midpricePredictionStart,
                yaxis='y',
                name=f"inaccurate lvl. 3 predictions",
                mode='markers',
                marker=dict(
                        color='#FF0000',
                        symbol='diamond',
                        size=5,
                        line=dict(
                            color='#DC143C',
                            width=3
                        )
                    )
            )

        ### prediction traces
        idx = list(np.where((DF_interp.iteration>start_index) & (DF_interp.iteration<end_index))[0])

        if len(idx) > 0:
            chunk_df_interp = DF_interp.iloc[idx[0]:idx[len(idx)-1]]

        for b in range(len(chunk_df_interp)):
            fig.add_shape(type="line",
                          x0=DF_interp.iteration.iloc[b],
                          y0=np.max(DF.midprice),
                          x1=DF_interp.iteration.iloc[b],
                          y1=np.min(DF.midprice),
                          line=dict(color="gray", width=0.5, dash="dash"))


        x_ticks = chunk_df['iteration'][::xTickSpacing]
        x_ticklabels = chunk_df['timestamp'][::xTickSpacing]

        fig.update_layout(
            xaxis=dict(tickvals=x_ticks, ticktext=x_ticklabels),
            title=f"ATTMO forecast {chunk_index+1}/{num_chunks}",
            xaxis_title="Time",
            yaxis_title="BTC/FDUSD",
            showlegend=True
        )

        img_filename = f"timeHorizon_{timeHorizon}_attmoForecastChunk_{chunk_index_str}_Of_{num_chunks_str}"
        filename = foldername_time_horizon + img_filename + ".html"

        chunk_index_str = "{:03d}".format(chunk_index+1)
        num_chunks_str = "{:03d}".format(num_chunks)
        plotly.offline.plot(fig, filename=filename,
                            image='png', image_filename=img_filename,
                            output_type='file',
                            validate=False)


def plotInOneImage(DF, DF_pred, DF_interp, timeHorizon, foldername_time_horizon, xTickSpacing):
    ### dcos traces
    ie_A = DF[abs(DF.currentEvent0) > 0]
    ie_A_dc = DF[abs(DF.currentEvent0) == 1]
    ie_A_os = DF[abs(DF.currentEvent0) == 2]
    ie_B = DF[abs(DF.currentEvent1) > 0]
    ie_B_dc = DF[abs(DF.currentEvent1) == 1]
    ie_B_os = DF[abs(DF.currentEvent1) == 2]
    ie_C = DF[abs(DF.currentEvent2) > 0]
    ie_C_dc = DF[abs(DF.currentEvent2) == 1]
    ie_C_os = DF[abs(DF.currentEvent2) == 2]

    ie_signal = DF[abs(DF.signalDetected) > 0]
    ie_signal_1 = DF[abs(DF.signalDetected) == 1]
    ie_signal_2 = DF[abs(DF.signalDetected) == 2]
    ie_signal_3 = DF[abs(DF.signalDetected) == 3]

    ### ATTMO forecast
    a = ie_signal.currentForecastLevel.values
    indices = []
    for i in range(1, len(ie_signal)):
        if a[i] != a[i - 1]:
            indices.append(i)

    ie_forecast = ie_signal.copy()
    ie_forecast = ie_forecast.iloc[indices]

    init_notebook_mode(connected=True)
    col_seq = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    yaxis = dict(
        title=f'test_image',
        showgrid=True,
        gridcolor='white',
        zeroline=False
    )

    ### axis 2
    #yaxis2 = dict(
    #    title="PNL (%)",
    #    side="right",
    #    overlaying="y",
    #    showgrid=True,
    #    gridcolor='lightgray',
    #    zeroline=False,
    #    #range=[0, 100],  # Set the range of the secondary y-axis
    #    tick0=0,  # Set the starting tick value
    #    dtick=0.1
    #)

    trace_mid = go.Scatter(
        x=DF.iteration,
        y=DF.midprice,
        yaxis='y',
        name="midprice",
        line=dict(color='gray', width=0.5)
    )


    traceA = go.Scatter(
        x=ie_A.iteration,
        y=ie_A.midprice,
        yaxis='y',
        name="delta interp. A",
        line=dict(color=col_seq[0], width=0.5)
    )
    traceA_dc = go.Scatter(
        x=ie_A_dc.iteration,
        y=ie_A_dc.midprice,
        name='delta interp. A DC',
        yaxis='y',
        mode='markers',
        marker=dict(
                color='white',
                symbol='square',
                size=2,
                line=dict(
                    color=col_seq[0],
                    width=0.5
                )
            )
    )
    traceA_os = go.Scatter(
        x=ie_A_os.iteration,
        y=ie_A_os.midprice,
        name='delta interp. A OS',
        yaxis='y',
        mode='markers',
        marker=dict(
                  color='black',
                  symbol='circle',
                  size=2,
                  line=dict(
                      color=col_seq[0],
                      width=0.5
                  )
              )
    )

    traceB = go.Scatter(
        x=ie_B.iteration,
        y=ie_B.midprice,
        yaxis='y',
        name="delta interp. B",
        line=dict(color=col_seq[1], width=1)
    )
    traceB_dc = go.Scatter(
        x=ie_B_dc.iteration,
        y=ie_B_dc.midprice,
        name='delta interp. B DC',
        yaxis='y',
        mode='markers',
        marker=dict(
                color='white',
                symbol='square',
                size=3,
                line=dict(
                    color=col_seq[1],
                    width=1
                )
            )
    )
    traceB_os = go.Scatter(
        x=ie_B_os.iteration,
        y=ie_B_os.midprice,
        name='delta interp. B OS',
        yaxis='y',
        mode='markers',
        marker=dict(
                  color='black',
                  symbol='circle',
                  size=3,
                  line=dict(
                      color=col_seq[1],
                      width=1
                  )
              )
    )

    traceC = go.Scatter(
        x=ie_C.iteration,
        y=ie_C.midprice,
        yaxis='y',
        name="delta interp. C",
        line=dict(color=col_seq[2], width=1.5)
    )
    traceC_dc = go.Scatter(
        x=ie_C_dc.iteration,
        y=ie_C_dc.midprice,
        name='delta interp. C DC',
        yaxis='y',
        mode='markers',
        marker=dict(
                color='white',
                symbol='square',
                size=4,
                line=dict(
                    color=col_seq[2],
                    width=1.5
                )
            )
    )
    traceC_os = go.Scatter(
        x=ie_C_os.iteration,
        y=ie_C_os.midprice,
        name='delta interp. C OS',
        yaxis='y',
        mode='markers',
        marker=dict(
                  color='black',
                  symbol='circle',
                  size=4,
                  line=dict(
                      color=col_seq[2],
                      width=1.5
                  )
              )
    )


    fig = go.Figure()
    fig.add_trace(trace_mid)
    fig.add_trace(traceA)
    fig.add_trace(traceA_dc)
    fig.add_trace(traceA_os)
    fig.add_trace(traceB)
    fig.add_trace(traceB_dc)
    fig.add_trace(traceB_os)
    fig.add_trace(traceC)
    fig.add_trace(traceC_dc)
    fig.add_trace(traceC_os)


    ### trend lines
    #idxStartRes = find_first_non_zero_indices(DF.resistanceLineFirstSample)
    idxLastRes = find_last_non_zero_indices(DF.resistanceLineLastSample)

    #idxStartSup = find_first_non_zero_indices(DF.supportLineFirstSample)
    idxLastSup = find_last_non_zero_indices(DF.supportLineLastSample)

    #ie_res_line_start = DF.iloc[idxStartRes]
    #ie_sup_line_start = DF.iloc[idxStartSup]
    ie_res_line_end = DF.iloc[idxLastRes]
    ie_sup_line_end = DF.iloc[idxLastSup]

    for i in range(len(ie_sup_line_end)):
        x_values = [ie_sup_line_end.supportLineFirstSample.iloc[i], ie_sup_line_end.supportLineLastSample.iloc[i]]
        y_values = [ie_sup_line_end.supportLineFirstMidprice.iloc[i], ie_sup_line_end.supportLineLastMidprice.iloc[i]]
        trace_sup_line = go.Scatter(x=x_values,
                                    y=y_values,
                                    yaxis='y',
                                    showlegend=False,
                                    line=dict(color='green', width=2.5))
        fig.add_trace(trace_sup_line)

    for i in range(len(ie_res_line_end)):
        x_values = [ie_res_line_end.resistanceLineFirstSample.iloc[i], ie_res_line_end.resistanceLineLastSample.iloc[i]]
        y_values = [ie_res_line_end.resistanceLineFirstMidprice.iloc[i], ie_res_line_end.resistanceLineLastMidprice.iloc[i]]
        trace_res_line = go.Scatter(x=x_values,
                                    y=y_values,
                                    yaxis='y',
                                    showlegend=False,
                                    line=dict(color='red', width=2.5))
        fig.add_trace(trace_res_line)


    a = ie_signal.currentForecastLevel.values
    indices = []
    for i in range(1, len(ie_signal)):
        if a[i] != a[i - 1]:
            indices.append(i)

    ie_forecast = ie_signal.copy()
    ie_forecast = ie_forecast.iloc[indices]

    for i in range(len(ie_forecast)-1):
        #if ie_signal.signalDetected.iloc[i] == -3:
        if ie_forecast.currentForecastLevel.iloc[i] == -3:
            fillcol = '#87CEFA'
        elif ie_forecast.currentForecastLevel.iloc[i] == -2:
            fillcol = '#4169E1'
        elif ie_forecast.currentForecastLevel.iloc[i] == -1:
            fillcol = '#191970'
        elif ie_forecast.currentForecastLevel.iloc[i] == 0:
            fillcol = '#F0F8FF'
        elif ie_forecast.currentForecastLevel.iloc[i] == 1:
            fillcol = '#FFFF66'
        elif ie_forecast.currentForecastLevel.iloc[i] == 2:
            fillcol = '#FFD700'
        elif ie_forecast.currentForecastLevel.iloc[i] == 3:
            fillcol = '#FFA500'

        fig.add_vrect(x0=ie_forecast.iteration.iloc[i], x1=ie_forecast.iteration.iloc[i+1],
                      annotation_text=str(ie_forecast.attmoForecast.iloc[i]), annotation_position="top left",
                      fillcolor=fillcol, opacity=0.25, line_width=0)


    ### accuracy trace
    ie_pred_correct = DF_pred.loc[DF_pred.predictionOutcome == 1]
    ie_pred_incorrect = DF_pred.loc[DF_pred.predictionOutcome == -1]


    ### signal level trace
    ie_pred_lvl_1 = DF_pred.loc[abs(DF_pred.signal) == 1]
    ie_pred_lvl_2 = DF_pred.loc[abs(DF_pred.signal) == 2]
    ie_pred_lvl_3 = DF_pred.loc[abs(DF_pred.signal) == 3]


    ### accuracy * level trace
    ie_pred_correct_lvl_1 = ie_pred_lvl_1.loc[ie_pred_lvl_1.predictionOutcome == 1]
    ie_pred_incorrect_lvl_1 = ie_pred_lvl_1.loc[ie_pred_lvl_1.predictionOutcome == -1]
    ie_pred_correct_lvl_2 = ie_pred_lvl_2.loc[ie_pred_lvl_2.predictionOutcome == 1]
    ie_pred_incorrect_lvl_2 = ie_pred_lvl_2.loc[ie_pred_lvl_2.predictionOutcome == -1]
    ie_pred_correct_lvl_3 = ie_pred_lvl_3.loc[ie_pred_lvl_3.predictionOutcome == 1]
    ie_pred_incorrect_lvl_3 = ie_pred_lvl_3.loc[ie_pred_lvl_3.predictionOutcome == -1]

    for p in range(len(ie_pred_incorrect_lvl_1)):
        fig.add_shape(
            type="rect",
            x0=ie_pred_incorrect_lvl_1.iterationPredictionStart.iloc[p],
            y0=ie_pred_incorrect_lvl_1.target.iloc[p],
            x1=ie_pred_incorrect_lvl_1.iterationPredictionEnd.iloc[p],
            y1=ie_pred_incorrect_lvl_1.stopLoss.iloc[p],
            line=dict(
                color="black",
                width=0.5,
            ),
            fillcolor = "red",
            opacity=0.2
        )
    for p in range(len(ie_pred_correct_lvl_1)):
        fig.add_shape(
            type="rect",
            x0=ie_pred_correct_lvl_1.iterationPredictionStart.iloc[p],
            y0=ie_pred_correct_lvl_1.target.iloc[p],
            x1=ie_pred_correct_lvl_1.iterationPredictionEnd.iloc[p],
            y1=ie_pred_correct_lvl_1.stopLoss.iloc[p],
            line=dict(
                color="black",
                width=0.5,
            ),
            fillcolor = "green",
            opacity=0.2
        )


    for p in range(len(ie_pred_incorrect_lvl_2)):
        fig.add_shape(
            type="rect",
            x0=ie_pred_incorrect_lvl_2.iterationPredictionStart.iloc[p],
            y0=ie_pred_incorrect_lvl_2.target.iloc[p],
            x1=ie_pred_incorrect_lvl_2.iterationPredictionEnd.iloc[p],
            y1=ie_pred_incorrect_lvl_2.stopLoss.iloc[p],
            line=dict(
                color="black",
                width=1,
            ),
            fillcolor = "red",
            opacity=0.2
        )
    for p in range(len(ie_pred_correct_lvl_2)):
        fig.add_shape(
            type="rect",
            x0=ie_pred_correct_lvl_2.iterationPredictionStart.iloc[p],
            y0=ie_pred_correct_lvl_2.target.iloc[p],
            x1=ie_pred_correct_lvl_2.iterationPredictionEnd.iloc[p],
            y1=ie_pred_correct_lvl_2.stopLoss.iloc[p],
            line=dict(
                color="black",
                width=1,
            ),
            fillcolor = "green",
            opacity=0.2
        )


    for p in range(len(ie_pred_incorrect_lvl_3)):
        fig.add_shape(
            type="rect",
            x0=ie_pred_incorrect_lvl_3.iterationPredictionStart.iloc[p],
            y0=ie_pred_incorrect_lvl_3.target.iloc[p],
            x1=ie_pred_incorrect_lvl_3.iterationPredictionEnd.iloc[p],
            y1=ie_pred_incorrect_lvl_3.stopLoss.iloc[p],
            line=dict(
                color="black",
                width=1.5,
            ),
            fillcolor = "red",
            opacity=0.4
        )
    for p in range(len(ie_pred_correct_lvl_3)):
        fig.add_shape(
            type="rect",
            x0=ie_pred_correct_lvl_3.iterationPredictionStart.iloc[p],
            y0=ie_pred_correct_lvl_3.target.iloc[p],
            x1=ie_pred_correct_lvl_3.iterationPredictionEnd.iloc[p],
            y1=ie_pred_correct_lvl_3.stopLoss.iloc[p],
            line=dict(
                color="black",
                width=1.5,
            ),
            fillcolor = "green",
            opacity=0.4
        )


    for b in range(len(DF_interp)):
        fig.add_shape(type="line",
                      x0=DF_interp.iteration.iloc[b],
                      y0=np.max(DF.midprice),
                      x1=DF_interp.iteration.iloc[b],
                      y1=np.min(DF.midprice),
                      line=dict(color="gray", width=0.5, dash="dash"))


    x_ticks = DF['iteration'][::xTickSpacing]
    x_ticklabels = DF['timestamp'][::xTickSpacing]


    fig.update_layout(
        xaxis=dict(tickvals=x_ticks, ticktext=x_ticklabels),
        title=f"ATTMO forecast {timeHorizon}",
        xaxis_title="Time",
        yaxis_title="BTC/FDUSD",
        showlegend=True
    )


    plotly.offline.plot(fig, filename=f"{foldername_time_horizon}attmo_forecast_{timeHorizon}.html",
                        image='png', image_filename=f"attmo_forecast_{timeHorizon}",
                        output_type='file',
                        validate=False)
