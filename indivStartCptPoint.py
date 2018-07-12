############################################################
## CONDUIT Lab                                            ##
## January 2018                                           ##
## Creator: Victoria Tolls                                ##
## indivStartCptPoint.py                                  ##
## Graphs each metric for each event as a bokeh line      ##
## graph add color to points if percentChange is selected ##
############################################################

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models import Span
from bokeh.transform import jitter, factor_cmap
from bokeh.palettes import mpl,brewer
import time
import datetime
import pandas as pd
import numpy as np
from pc_hrv import getBeforeDF,getAverageBaselineDF,getLabeledPercentChangeDF


''' Line plot of a timeseries. Takes in dictionary of data. Option to choose percentChange calculation
    to color points based on 25% threshold. Return bokeh figure. '''
## MAIN FUNCTION ##
def indivStartCptPoint(dict_data,title,percentChange=True):
    palette = brewer["Set1"][3] #colors for plotting
    #get the starttime and the changepoint time from a string
    s_time = 0.0
    #if percent change is selected, update data with colors
    if percentChange:
        source = calcPercentChange(dict_data,s_time)
    else:
        source = calcPercentChange(dict_data,s_time,lst_col=(['SlateGray'] * len(dict_data["indexed"])))
    plot_tools = {'tools':['reset','save']}
    #create bokeh figure
    p1 = figure(width=400, plot_height=210,title=title,**plot_tools)
    #graph the series
    p1.circle('indexed', 'value', size=10, alpha=0.9,line_color=None, fill_color='color', source=source)
    p1.line('indexed', 'value',source=source,line_color="DarkSlateGray")
    #add vertical lines at start and chanepoints
    time_vline = Span(location=0, dimension='height', line_color=palette[0], line_width=1)
    cpt_vline = Span(location=dict_data["time_diff_change"], dimension='height', line_color=palette[1], line_width=1)
    p1.renderers.extend([time_vline, cpt_vline])
    p1.toolbar.logo = None
    return p1

#---------------------------------------------------------------------------------------------#
def calcPercentChange(dict_data,s_time,lst_col=[]):
    if not lst_col:
        #dataframe to calculate colors
        df = pd.DataFrame(columns=["time", "value"])
        #convert to datetime from string
        df["time"] = dict_data["indexed"]
        df["value"] = dict_data["value"]
        temp = getBeforeDF(df,s_time,10)
        temp2 = getAverageBaselineDF(temp)
        temp3,vals = getLabeledPercentChangeDF(df,temp2,25)
        #replace percent, change with colours
        lst_col = list(temp3["value"])
        lst_col = ['Lime' if x=='significant' else x for x in lst_col]
        lst_col = ['DarkSlateGray' if x=='baseline' else x for x in lst_col]
        final_df = pd.DataFrame(columns=["indexed", "value"])
        final_df["indexed"] = dict_data["indexed"]
        final_df["value"] = dict_data["value"]
        final_df["color"] = lst_col
        final_df = final_df.dropna(how="any")
        final_df = final_df.reset_index()
        source = ColumnDataSource(final_df)
    else:
        final_df = pd.DataFrame(columns=["indexed", "value"])
        final_df["indexed"] = dict_data["indexed"]
        final_df["value"] = dict_data["value"]
        final_df["color"] = lst_col
        source = ColumnDataSource(final_df)
    return source