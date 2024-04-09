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


date = '08-04-2024_19-29-27' # '08-04-2024_16-20-45'
#'08-04-2024_16-08-15' #'08-04-2024_14-41-54' #'07-04-2024_14-53-24' #'08-04-2024_09-48-15' #'05-04-2024_11-48-51
### load signal detection files

#foldername = f"C:/Users/thoma/Documents/ATTMO/BTCFDUSD_{date}/"


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




def runPostprocess(date):
    foldername = f"C:/Users/thoma/Documents/BTCFDUSD_data/BTCFDUSD_{date}/"

    results_DF = pd.DataFrame(columns = ['timeHorizon', 'forecast_duration_X', 'forecast_duration_SD', 'pred_duration_X', 'pred_duration_SD',
                                        'overall_n_pred', 'n_lvl_1_pred', 'n_lvl_2_pred', 'n_lvl_3_pred',
                                        'overall_accuracy', 'pred_accuracy_lvl_1', 'pred_accuracy_lvl_2', 'pred_accuracy_lvl_3'])

    for t in range(len(config.timeHorizons)):
        chunk_size = config.blockLengths*10

        foldername_time_horizon = foldername+config.timeHorizons[t]+"/"
        #foldername_interpolation = foldername_time_horizon+"interpolation/"
        foldername_signal_detector = foldername_time_horizon+"signal_detector/"
        foldername_predictions = foldername_time_horizon+"predictions/"

        file_path = foldername_time_horizon + f"DF_signals_{config.timeHorizons[t]}.csv"
        if os.path.exists(file_path):
            print("Signal detector file exists. Loading...")
            DF = pd.read_csv(foldername_time_horizon + f"DF_signals_{config.timeHorizons[t]}.csv")
        else:
            print("Creating signal detector file...")
            event_files = glob.glob(f"{foldername_signal_detector}*.parquet")
            for i in range(len(event_files)):
                if i == 0:
                    DF = pd.read_parquet(event_files[i])
                elif i > 0:
                    df = pd.read_parquet(event_files[i])
                    DF = pd.concat([DF, df])
            DF.to_csv(foldername_time_horizon + f"DF_signals_{config.timeHorizons[t]}.csv")


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


        ### trend lines
        idxStartRes = find_first_non_zero_indices(DF.resistanceLineFirstSample)
        idxLastRes = find_last_non_zero_indices(DF.resistanceLineLastSample)

        idxStartSup = find_first_non_zero_indices(DF.supportLineFirstSample)
        idxLastSup = find_last_non_zero_indices(DF.supportLineLastSample)

        IE_res_line_start = DF.iloc[idxStartRes]
        IE_sup_line_start = DF.iloc[idxStartSup]
        IE_res_line_end = DF.iloc[idxLastRes]
        IE_sup_line_end = DF.iloc[idxLastSup]


        ### ATTMO forecast
        a = ie_signal.currentForecastLevel.values
        indices = []
        for i in range(1, len(ie_signal)):
            if a[i] != a[i - 1]:
                indices.append(i)

        ie_forecast = ie_signal.copy()
        ie_forecast = ie_forecast.iloc[indices]


        ### load prediction files
        foldername_predictions = foldername_time_horizon+"predictions/"

        file_path = foldername_time_horizon + f"DF_predictions_{config.timeHorizons[t]}.csv"
        if os.path.exists(file_path):
            print("Predictions file exists. Loading...")
            DF_pred = pd.read_csv(foldername_time_horizon + f"DF_predictions_{config.timeHorizons[t]}.csv")
        else:
            print("Creating predictions file...")
            event_files = glob.glob(f"{foldername_predictions}*.parquet")
            for i in range(len(event_files)):
                if i == 0:
                    DF_pred = pd.read_parquet(event_files[i])
                else:
                    df_pred = pd.read_parquet(event_files[i])
                    DF_pred = pd.concat([DF_pred, df_pred])
            DF_pred.to_csv(foldername_time_horizon + f"DF_predictions_{config.timeHorizons[t]}.csv")


        ### accuracy trace
        IE_pred_correct = DF_pred.loc[DF_pred.predictionOutcome == 1]
        IE_pred_incorrect = DF_pred.loc[DF_pred.predictionOutcome == -1]


        ### signal level trace
        IE_pred_correct_lvl_1 = IE_pred_correct.loc[IE_pred_correct.signal == 1]
        IE_pred_incorrect_lvl_1 = IE_pred_incorrect.loc[IE_pred_incorrect.signal == -1]

        IE_pred_correct_lvl_2 = IE_pred_correct.loc[IE_pred_correct.signal == 2]
        IE_pred_incorrect_lvl_2 = IE_pred_incorrect.loc[IE_pred_incorrect.signal == -2]

        IE_pred_correct_lvl_3 = IE_pred_correct.loc[IE_pred_correct.signal == 3]
        IE_pred_incorrect_lvl_3 = IE_pred_incorrect.loc[IE_pred_incorrect.signal == -3]


        ### descriptives
        forecast_duration = [0]
        for i in range(len(ie_forecast)):
            forecast_duration.append(ie_forecast.iteration.iloc[i] - forecast_duration[i])
        forecast_durations = forecast_duration[1:]
        forecast_duration_X = np.round(np.mean(forecast_durations),2)
        forecast_duration_SD = np.round(np.std(forecast_durations),2)

        pred_duration_X = np.round(np.mean(np.array(DF_pred.iterationPredictionEnd)-np.array(DF_pred.iterationPredictionStart)))
        pred_duration_SD = np.round(np.mean(np.array(DF_pred.iterationPredictionEnd)-np.array(DF_pred.iterationPredictionStart)))

        overall_n_pred = len(IE_pred_correct)+len(IE_pred_incorrect)
        n_lvl_1_pred = len(IE_pred_correct_lvl_1)+len(IE_pred_incorrect_lvl_1)
        n_lvl_2_pred = len(IE_pred_correct_lvl_2)+len(IE_pred_incorrect_lvl_2)
        n_lvl_3_pred = len(IE_pred_correct_lvl_3)+len(IE_pred_incorrect_lvl_3)

        overall_accuracy = np.round(len(IE_pred_correct) * 100 / (len(IE_pred_correct)+len(IE_pred_incorrect)),2)

        pred_accuracy_lvl_1 = 0
        pred_accuracy_lvl_2 = 0
        pred_accuracy_lvl_3 = 0

        if n_lvl_1_pred > 0:
            pred_accuracy_lvl_1 = np.round(len(IE_pred_correct_lvl_1) * 100 / (len(IE_pred_correct_lvl_1)+len(IE_pred_incorrect_lvl_1)),2)
        if n_lvl_2_pred > 0:
            pred_accuracy_lvl_2 = np.round(len(IE_pred_correct_lvl_2) * 100 / (len(IE_pred_correct_lvl_2)+len(IE_pred_incorrect_lvl_2)),2)
        if n_lvl_3_pred > 0:
            pred_accuracy_lvl_3 = np.round(len(IE_pred_correct_lvl_3) * 100 / (len(IE_pred_correct_lvl_3)+len(IE_pred_incorrect_lvl_3)),2)


        print("")
        print(f"{config.timeHorizons[t]}:")
        print(f"Mean forecast duration = {np.round(forecast_duration_X/60)} min. (SD = {np.round(forecast_duration_SD/60)}).")
        print(f"Mean prediction duration = {np.round(pred_duration_X/60)} min. (SD = {np.round(pred_duration_SD/60)}).")
        print(f"Tot predictions generated = {overall_n_pred}: {n_lvl_1_pred} lvl. 1, {n_lvl_2_pred} lvl. 2, and {n_lvl_3_pred} lvl. 3.")
        print(f"Overall accuracy = {overall_accuracy} %.")
        print(f"Accuracy lvl. 1 = {pred_accuracy_lvl_1} %.")
        print(f"Accuracy lvl. 2 = {pred_accuracy_lvl_2} %.")
        print(f"Accuracy lvl. 3 = {pred_accuracy_lvl_3} %.")

        results_DF.loc[t] = [config.timeHorizons[t], forecast_duration_X, forecast_duration_SD,
                            pred_duration_X, pred_duration_SD,
                            overall_n_pred, n_lvl_1_pred, n_lvl_2_pred, n_lvl_3_pred,
                            overall_accuracy, pred_accuracy_lvl_1, pred_accuracy_lvl_2, pred_accuracy_lvl_3]



        if chunk_size > 0:
            num_chunks = len(DF) // chunk_size + (1 if len(DF) % chunk_size != 0 else 0)  # Calculate the number of chunks

            for chunk_index in range(num_chunks):
                start_index = chunk_index * chunk_size
                end_index = min((chunk_index + 1) * chunk_size, len(DF))

                chunk_df = DF.iloc[start_index:end_index]


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
                ie_signal = chunk_df[abs(chunk_df.signalDetected) > 0]
                ie_signal_1 = chunk_df[abs(chunk_df.signalDetected) == 1]
                ie_signal_2 = chunk_df[abs(chunk_df.signalDetected) == 2]
                ie_signal_3 = chunk_df[abs(chunk_df.signalDetected) == 3]


                ### trend lines
                idxStartRes = find_first_non_zero_indices(chunk_df.resistanceLineFirstSample)
                idxLastRes = find_last_non_zero_indices(chunk_df.resistanceLineLastSample)
                idxStartSup = find_first_non_zero_indices(chunk_df.supportLineFirstSample)
                idxLastSup = find_last_non_zero_indices(chunk_df.supportLineLastSample)

                IE_res_line_start = chunk_df.iloc[idxStartRes]
                IE_sup_line_start = chunk_df.iloc[idxStartSup]
                IE_res_line_end = chunk_df.iloc[idxLastRes]
                IE_sup_line_end = chunk_df.iloc[idxLastSup]



                ### create figure
                fn = f"attmo_forecast_{config.timeHorizons[t]}"
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



                for i in range(len(IE_sup_line_end)):
                    x_values = [IE_sup_line_end.supportLineFirstSample.iloc[i], IE_sup_line_end.supportLineLastSample.iloc[i]]
                    y_values = [IE_sup_line_end.supportLineFirstMidprice.iloc[i], IE_sup_line_end.supportLineLastMidprice.iloc[i]]
                    trace_sup_line = go.Scatter(x=x_values,
                                                y=y_values,
                                                yaxis='y',
                                                showlegend=False,
                                                line=dict(color='green', width=3))
                    fig.add_trace(trace_sup_line)

                for i in range(len(IE_res_line_end)):
                    x_values = [IE_res_line_end.resistanceLineFirstSample.iloc[i], IE_res_line_end.resistanceLineLastSample.iloc[i]]
                    y_values = [IE_res_line_end.resistanceLineFirstMidprice.iloc[i], IE_res_line_end.resistanceLineLastMidprice.iloc[i]]
                    trace_res_line = go.Scatter(x=x_values,
                                                y=y_values,
                                                yaxis='y',
                                                showlegend=False,
                                                line=dict(color='red', width=3))
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
                    IE_pred_correct = chunk_df_pred.loc[chunk_df_pred.predictionOutcome == 1]
                    IE_pred_incorrect = chunk_df_pred.loc[chunk_df_pred.predictionOutcome == -1]


                    ### signal level * accuracy trace
                    IE_pred_correct_lvl_1 = IE_pred_correct.loc[IE_pred_correct.signal == 1]
                    IE_pred_incorrect_lvl_1 = IE_pred_incorrect.loc[IE_pred_incorrect.signal == -1]
                    IE_pred_correct_lvl_2 = IE_pred_correct.loc[IE_pred_correct.signal == 2]
                    IE_pred_incorrect_lvl_2 = IE_pred_incorrect.loc[IE_pred_incorrect.signal == -2]
                    IE_pred_correct_lvl_3 = IE_pred_correct.loc[IE_pred_correct.signal == 3]
                    IE_pred_incorrect_lvl_3 = IE_pred_incorrect.loc[IE_pred_incorrect.signal == -3]



                    trace_pred_correct_lvl_1 = go.Scatter(
                        x=IE_pred_correct_lvl_1.iterationPredictionStart,
                        y=IE_pred_correct_lvl_1.midpricePredictionStart,
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
                        x=IE_pred_incorrect_lvl_1.iterationPredictionStart,
                        y=IE_pred_incorrect_lvl_1.midpricePredictionStart,
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
                        x=IE_pred_correct_lvl_2.iterationPredictionStart,
                        y=IE_pred_correct_lvl_2.midpricePredictionStart,
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
                        x=IE_pred_incorrect_lvl_2.iterationPredictionStart,
                        y=IE_pred_incorrect_lvl_2.midpricePredictionStart,
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
                        x=IE_pred_correct_lvl_3.iterationPredictionStart,
                        y=IE_pred_correct_lvl_3.midpricePredictionStart,
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
                        x=IE_pred_incorrect_lvl_3.iterationPredictionStart,
                        y=IE_pred_incorrect_lvl_3.midpricePredictionStart,
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



                x_ticks = chunk_df['iteration'][::600]  # Extract ticks every 600 iterations
                x_ticklabels = chunk_df['timestamp'][::600]  # Extract labels every 600 iterations

                fig.update_layout(
                    xaxis=dict(tickvals=x_ticks, ticktext=x_ticklabels),
                    title=f"ATTMO forecast {chunk_index+1}/{num_chunks}",
                    xaxis_title="Time",
                    yaxis_title="BTC/FDUSD",
                    showlegend=True
                )


                chunk_index_str = "{:03d}".format(chunk_index+1)
                num_chunks_str = "{:03d}".format(num_chunks)
                plotly.offline.plot(fig, filename=f"attmo_forecast_chunk_{chunk_index_str}_of_{num_chunks_str}.html",
                                    image='png', image_filename=f"attmo_forecast_chunk_{chunk_index_str}_of_{num_chunks_str}",
                                    output_type='file',
                                    validate=False)


    results_DF.to_csv(foldername + f"descriptives.csv")
    return results_DF
