############################################################
## CONDUIT Lab                                            ##
## January 2018                                           ##
## Creator: Victoria Tolls                                ##
## callbacks.py                                           ##
## Functionality of button callbacks, called from main.py ##
## contains code to load the appropriate graphs.          ##
## Uses helpers.py                                        ##
############################################################

from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS, Jitter, Div
from bokeh.models.widgets import DataTable, TableColumn, RadioButtonGroup, PreText, RangeSlider, CheckboxButtonGroup
from bokeh.models.widgets import Button, Tabs, Panel, MultiSelect, HTMLTemplateFormatter,TextInput,Toggle
from bokeh.layouts import widgetbox, layout,gridplot
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.events import ButtonClick
from bokeh.palettes import Spectral

from stackedBarChart import stackedBarChartMultiple
from dotBoxplot import graphDotBoxplot
from indivStartCptPoint import indivStartCptPoint
from heatmap import heatmap
from kmeansClustering import kmeansClustering

import numpy as np
import pandas as pd
from os.path import dirname, join
import scipy


from helpers import *

''' Update Button passed in with new source for callback (saving) by going through new ids and geting value from source
    Return a pandas dataframe of the selected data   '''
def dataSelectCallback(btn_save_selected,unq_sel,source_stats):
    #create dataframe from the table data
    newdf = pd.DataFrame(source_stats.data,index=source_stats.data["index"])
    #put selected indexes on top and the rest on the bottom
    top = newdf[newdf['index'].isin(unq_sel)]
    #update button callback
    source_sel = ColumnDataSource(top)
    line = ColumnDataSource(data = dict(start=['label,index,daySOFA,APACHEII,sex,type,vaso_type,vitalstatday28\n']))
    btn_save_selected.callback = CustomJS(args=dict(source=source_sel,file_text=line),
                        code=open(join(dirname(__file__), "download.js")).read())
    return top

''' Load the initial statistics graphs, get all event ids from the graph source
    Return list of new graphs and the associated sources   '''
def loadInitalStatsGraphs(source,btn_save_selected,source_stats,clster_col):
    unq_sel = [ float(x) for x in list(set(source.data['index'])) ]
    selected_df = dataSelectCallback(btn_save_selected,unq_sel,source_stats)
    new_graphs,graph_stats = graphStats(selected_df,clster_col)
    return new_graphs,graph_stats

''' Using graph source, get selected data, update label in source_stats for the selected data   '''
def labelSelected(source,source_stats,txt_label):
    label = txt_label.value
    data = source.data
    if source.selected != []:
        selected = source.selected['1d']['indices']
        sel_ids = [ float(data['index'][x]) for x in selected ]
        unq_sel = list(set(sel_ids))
        sel_ind = [ source_stats.data['index'].index(elm) for elm in unq_sel ]
        for ind in sel_ind:
            source_stats.data['label'][ind] = label
        txt_label.value = ""

''' Take the original data and label the clusters based on the data. '''
def labelOriginalClustersHeatmap(source,source_stats,clusters):
    for key, value in clusters.items():
        for elm in value:
            for i in range(len(source_stats.data["index"])):
                if float(elm) == float(source_stats.data["index"][i]):
                    source_stats.data["label"][i] = key
                    break
            

''' When data is selected in source, update the data in source_stats giving the Button a new source to save '''
def onChangeSelectCallback(source,new,btn_save_selected,source_stats,graph_stats,clstr_col):
    typelst=["histo","histo","histo","histo","cate", "cate", "cate","cate"]
    data = source.data
    selected = new['1d']['indices']
    if selected == []: #if none are selected use entire dataset
        sel_ids = [ float(x) for x in list(data['index']) ]
    else:
        sel_ids = [ float(data['index'][x]) for x in selected ]
    unq_sel = list(set(sel_ids))
    selected_df = dataSelectCallback(btn_save_selected,unq_sel,source_stats)
    if clstr_col == "type":
        typelst=["histo","histo","histo","histo","histo", "cate", "cate","cate"]
        for i in range(0,len(typelst)):
            min_df = selected_df[["lactate_clearance","APACHEII","daySOFA","pressor_restarted",
                    "sex","vaso_type","vitalstatday28","index","label","type"]]
            updateStats(graph_stats[i],typelst[i],min_df,min_df.columns[i],unq_sel,clstr_col)
    else:
        for i in range(0,len(typelst)):
            min_df = selected_df[["lactate_clearance","APACHEII","daySOFA","pressor_restarted",
                    "sex","type","vaso_type","vitalstatday28","index","label"]]
            updateStats(graph_stats[i],typelst[i],min_df,min_df.columns[i],unq_sel,clstr_col)

''' Get event dicts from list of dicts based on ids from events, using speicified metrics and minutes selected, loop through 
    event dicts and boxplot the data
    Return list of boxplot bokeh figures  '''
def clickBtnBoxPlot(dicts,events, metrics,minutes,waveform,colIndvPlots): 
    #get events from the dicts that match the selected events and have HRV metrics 
    results = [evnt for evnt in dicts if evnt['_id'] in [float(x) for x in events]]
    results = [evnt for evnt in results if bool(evnt[waveform][minutes])] #"HRVMetrics"
    #load the pandas dataframe holding the data for the boxplot
    dict_source = load_figure_source(metrics,results,minutes,waveform)
    #create column source
    source = ColumnDataSource(dict_source)
    source_stats = ColumnDataSource(dict_source)
    #save button, one for all data, one for selected data
    btn_save_selected = Button(label="Download", button_type="success",width=150)
    btn_label_selected = Button(label="Label Selected", button_type="success", width=150)
    txt_input_label = TextInput(width=150,height=10)
    btnIndivPlots = Button(button_type="warning", label="Selected Individual Plots", sizing_mode="fixed", width=200)
    #load stats graphs
    new_graphs, graph_stats = loadInitalStatsGraphs(source,btn_save_selected,source_stats,'type')
    #Callback for selecting data (when data is selected update the stats graphs (row))
    def onChangeSelect(attr,old,new):
        onChangeSelectCallback(source,new,btn_save_selected,source_stats,graph_stats,'type')
    def onClickLabel():
        labelSelected(source,source_stats,txt_input_label)
        onChangeSelectCallback(source,source.selected,btn_save_selected,source_stats,graph_stats,'type')
    def clickBtnIndivGraphs():
        getIndividualPlots(dicts,source,metrics,minutes,colIndvPlots,waveform)
    #assign callback
    source.on_change('selected',onChangeSelect)
    btn_label_selected.on_click(onClickLabel) #add callback
    btnIndivPlots.on_click(clickBtnIndivGraphs)
    lstBoxplots = []
    #go through all metrics and create bokeh figure using graphDotBoxplot function
    for metric in metrics:
        df = pd.DataFrame(source.data)
        #magnitude change graph
        pvalue = getKruskalWallis(df,'type',(metric+'_mag'))
        p1 = graphDotBoxplot(source, (metric+'_mag'), 'type', metric+': Magnitude of changepoint (p:'+str(pvalue)+')', 'Event Type', 'Magnitude (delta HRV)')
        pvalue = getKruskalWallis(df,'type',(metric+'_timediff'))
        #time diff plot
        p2 = graphDotBoxplot(source, (metric+'_timediff'), 'type', metric+': Event-changepoint time (p:'+str(pvalue)+')', 'Event Type', 'Time (minutes)')
        txt = PreText(text=metric, width=150,height=20)
        lstBoxplots.append(row(txt,p1,p2))
    clusterGraphs = getClusterCompStats(source_stats,'type')
    plot_width = clusterGraphs.children[1].children[0].plot_width
    lstNewGraphs = [ layout([[widgetbox(btnIndivPlots,btn_save_selected,btn_label_selected,txt_input_label,width=210),
            lstBoxplots,
            widgetbox(Div(text="""<hr width='"""+str((plot_width*2)+40)+"""' size="1800" >""",style={'border-color':"#D5D8DC",'z-index': '-1'}),width=30),
            clusterGraphs,new_graphs],])
    ]
    return lstNewGraphs

''' Get event dicts from list of dicts based on ids from events, using speicified metrics and minutes selected, loop through 
    event dicts and heatmap the data based on the number of n_clusters specified
    Return list of heatmap bokeh figures  '''
def clickBtnHeatmap(dicts,events, metrics,minutes,waveform,n_clusters,colIndvPlots):
    #get events from list of events, that matches the id's of the selected events
    results = [evnt for evnt in dicts if evnt['_id'] in [float(x) for x in events]]
    results = [evnt for evnt in results if bool(evnt[waveform][minutes])]
    add_dataf = load_figure_source(metrics,results,minutes,waveform)
    source_stats = ColumnDataSource(add_dataf)
    #save button, one for all data, one for selected data
    btn_save_selected = Button(label="Download", button_type="success",width=150)
    btn_label_selected = Button(label="Label Selected", button_type="success", width=150)
    txt_input_label = TextInput(width=150,height=30)
    btnIndivPlots = Button(button_type="warning", label="Selected Individual Plots", sizing_mode="fixed", width=200)
    hmap = None
    #loop through selected metrics to create heatmaps
    for metric in metrics:
        lstValues = []
        #list of event id's as strings
        lstEvents = [str(evnt["_id"]) for evnt in results]
        #count number of time points for each metric, get the minimum
        numobs = min([len(evnt[waveform][minutes][metric]["value"]) for evnt in results if metric in evnt[waveform][minutes].keys()])
        #create pandas dataframe as a source, (nxm) n events with m time observations, full of NaN until filled
        df = pd.DataFrame(index=lstEvents,columns=[metric+"_"+str(x) for x in list(range(0,numobs))])
        #fill df with values, for each event (index) fill columns with time values
        for ind in lstEvents:
            lstValues = [ evnt[waveform][minutes][metric]["value"] for evnt in results if str(evnt["_id"]) == ind if metric in evnt[waveform][minutes].keys() ]
            df.loc[ind] = lstValues[0][0:numobs]
        #heatmap figure
        hmap,source,clusters = heatmap(df,n_clusters+1,title=metric) #add one to clustering to allow for the upper cluster
        #function to change things...
        def onChangeSelect(attr,old,new):
            onChangeSelectCallback(source,new,btn_save_selected,source_stats,graph_stats,'label')
        def onClickLabel():
            labelSelected(source,source_stats,txt_input_label)
            onChangeSelectCallback(source,source.selected,btn_save_selected,source_stats,graph_stats,'label')
        def clickBtnIndivGraphs():
            getIndividualPlots(dicts,source,metrics,minutes,colIndvPlots,waveform)
        source.on_change('selected',onChangeSelect)
        btn_label_selected.on_click(onClickLabel) #add callback
        btnIndivPlots.on_click(clickBtnIndivGraphs)
        #append graph to list
        labelOriginalClustersHeatmap(source,source_stats,clusters)
        #create boxplot layout for the stats from the source_stats (comparing clusters)
    #load stats graphs
    new_graphs, graph_stats = loadInitalStatsGraphs(source,btn_save_selected,source_stats,'label')
    clusterGraphs = getClusterCompStats(source_stats,'label')
    plot_width = clusterGraphs.children[1].children[0].plot_width
    lstNewGraphs = [layout([[widgetbox(btnIndivPlots,btn_save_selected,btn_label_selected,txt_input_label,width=210), 
            hmap,
            widgetbox(Div(text="""<hr width='"""+str((plot_width*2)+40)+"""' size="2100" >""",style={'border-color':"#D5D8DC",'z-index': '-1'}),width=30),
            clusterGraphs,new_graphs],])]
    return lstNewGraphs

''' Get event dicts from list of dicts based on ids from events, using speicified metrics and minutes selected, loop through 
    event dicts and heatmap the data based on the number of n_clusters specified
    Data is loaded differently based on the "type" of data specified from bokeh option list
    Return list of kmeansclustering bokeh figures  '''
def clickBtnKMeansClustering(dicts, events, metrics, minutes, waveform, type,n_clusters,colIndvPlots):
    #loop through and get the events that match the selected events ids
    results = [evnt for evnt in dicts if evnt['_id'] in [float(x) for x in events]]
    results = [evnt for evnt in results if bool(evnt[waveform][minutes])]
    lstEvents = [str(evnt["_id"]) for evnt in results]#list of event ids as strings
    lstDFs = [] #empty list for dataframes to concatenate 
    title =""
    add_dataf = load_figure_source(metrics,results,minutes,waveform)
    source_stats = ColumnDataSource(add_dataf)
    #load widgets
    btn_save_selected = Button(label="Download", button_type="success",width=150)
    btn_label_selected = Button(label="Label Selected", button_type="success", width=150)
    txt_input_label = TextInput(width=150,height=30)
    btnIndivPlots = Button(button_type="warning", label="Selected Individual Plots", sizing_mode="fixed", width=200)
    for metric in metrics:
        #get the number of time points
        numobs = min([len(evnt[waveform][minutes][metric]["value"]) for evnt in results if metric in list(evnt[waveform][minutes].keys()) ])
        #default using all timepoints for all metrics
        start = 0
        end = numobs
        title = str(int(minutes)*2)+" min surrounding event: "
        if type == 1: #using only time after the start point (ie. middle to end of list)
            start = (numobs//2)
            end = numobs
            title = minutes+" min after event: "
        elif type == 2: #using only time before start point
            start = 0
            end = (numobs//2)
            title = minutes+" min before event: "
        elif type == 3: #first 5 minutes
            start = (numobs//2)
            end = start+1
            title = "First 5 min after event: "
        #create empty (NaN) data frame with nxm structure, n events (ids,rows) with m columns (time points)
        df = pd.DataFrame(index=lstEvents,columns=[metric+"_"+str(x) for x in list(range(start,end))])
        #fill empty data frame, at each index (id) with the m time points
        for ind in lstEvents:
            lstValues = [evnt[waveform][minutes][metric]["value"] for evnt in results if str(evnt["_id"]) == ind and metric in list(evnt[waveform][minutes].keys())]
            if lstValues != []:
                df.loc[ind] = lstValues[0][start:end]
            else:
                df.loc[ind] = [np.nan]*(end-start)
        #create list of dataframes
        lstDFs.append(df)
    results_df = pd.concat(lstDFs,axis=1, join="inner") #concatenate the dataframes
    results_df.index = lstEvents #ensure that the index is the event ids
    title = title + ', '.join(metrics) #create title
    #call kmeansClustering function, and return results as list for use on curdoc()
    kmcluster,source,clusters = kmeansClustering(results_df,n_clusters,title)
    #select data callback so that stats graphs only have selected data
    def onChangeSelect(attr,old,new):
        onChangeSelectCallback(source,new,btn_save_selected,source_stats,graph_stats,'label')
    def onClickLabel():
        labelSelected(source,source_stats,txt_input_label)
        onChangeSelectCallback(source,source.selected,btn_save_selected,source_stats,graph_stats,'label')
    def clickBtnIndivGraphs():
        getIndividualPlots(dicts,source,metrics,minutes,colIndvPlots,waveform)
    #label source stats with orginal cluster labels
    source_stats.data["label"] = [ str(x) for x in list(clusters["color"]) ]
    source.on_change('selected',onChangeSelect)
    btn_label_selected.on_click(onClickLabel) #add callback
    btnIndivPlots.on_click(clickBtnIndivGraphs)
    #load stats graphs
    new_graphs, graph_stats = loadInitalStatsGraphs(source,btn_save_selected,source_stats,'label')
    clusterGraphs = getClusterCompStats(source_stats,'label')
    plot_width = clusterGraphs.children[1].children[0].plot_width
    lstNewGraphs = [layout([[widgetbox(btnIndivPlots,btn_save_selected,btn_label_selected,txt_input_label,width=210),
            kmcluster,
            widgetbox(Div(text="""<hr width='"""+str((plot_width*2)+40)+"""' size="2100" >""",style={'border-color':"#D5D8DC",'z-index': '-1'}),width=30),#Div(text="""<div class="vl"> </div>""",height=500,width=50),
            clusterGraphs,new_graphs],])
    ]
    return lstNewGraphs

''' Plot individual event timeseries, selected events, annotate with changepoint and event start '''
def getIndividualPlots(dicts,source,metrics,minutes,colgraphs,waveform):
    lstNewGraphs = []
    data = source.data
    selected = source.selected['1d']['indices']
    if selected == []: #if none are selected use entire dataset
        sel_ids = [ float(x) for x in list(data['index']) ]
    else:
        sel_ids = [ float(data['index'][x]) for x in selected ]
    unq_sel = list(set(sel_ids))
    #loop through all metrics, each metric is a row in the figure layout
    for metric in metrics:
        #get events based on selected data from boxplot (table) based on lsit of event dicts
        results = [evnt for evnt in dicts if evnt['_id'] in [float(x) for x in unq_sel]]
        results = [evnt for evnt in results if bool(evnt[waveform][minutes])]
        #for each event, graph the individual plot, using indivStartCptPoint function
        for evnt in results:
            lstNewGraphs.append(indivStartCptPoint(evnt[waveform][minutes][metric], metric+" "+evnt["event_type"]+" "+str(evnt["_id"])))
    colgraphs.children = [gridplot(lstNewGraphs,ncols=4,merge_tools=False)]