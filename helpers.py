############################################################
## CONDUIT Lab                                            ##
## January 2018                                           ##
## Creator: Victoria Tolls                                ##
## helpers.py                                             ##
## Helper functions, used in main.py and callbacks.py     ##
############################################################

from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS, Jitter, Div
from bokeh.models.widgets import DataTable, TableColumn, RadioButtonGroup, PreText, RangeSlider, CheckboxButtonGroup
from bokeh.models.widgets import Button, Tabs, Panel, MultiSelect
from bokeh.layouts import widgetbox, layout, gridplot
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.events import ButtonClick
from bokeh.palettes import Spectral
from bokeh.palettes import d3,mpl,brewer
from bokeh.models import Selection

from admitDiagnosisRetrival import admitDiagnosisRetrieval
import numpy as np
import pandas as pd
import scipy

from dotBoxplot import graphDotBoxplot
from stackedBarChart import stackedBarChart,stackedBarChartMultiple

from os.path import dirname, join

''' Load the inital dataset from json file into list of dicts
    Return list of dictionaries '''
def loadJsonData():
    import json
    dicts = []
    d = json.load(open(join(dirname(__file__), "data/event_based_dict_storage_ALL_NOTIME_FINAL.json")))
    for key in d['_default'].keys():
        dicts.append(d['_default'][key])
    return dicts

''' Load the event list, for use in Bokeh mulsitselect, loop through dict list and get id for each event
    Return list of [(label,id)] and list of active [] '''
def loadEventOptions(lstEvents):
    lstOptions = []
    lstActive = []
    for e in lstEvents:
        type = 'None' if e["type"] is None else e["type"]
        opt = str(e["event_type"]+" | "+type+" | "+str(e['_id']))
        id = str(e["_id"])
        lstActive.append(id)
        lstOptions.append((id,opt))
    return lstOptions,lstActive

''' Load the metric list, go through dict events and get all possible metrics, minimize and append to list of options (sorted)
    Return list of [metric_name] '''
def loadMetrics(lstEvents,waveform,minutes):
    if waveform == "HRVMetrics": #hardcoded, only use a sub-set of HRV metrics, these most events contain
        opts = ['HFPowerLombScargle', 'MultiFractal_c2', 'MultiscaleEntropy', 'MultiFractal_c1', 
            'PowerLawSlopeLombScargle', 'eScaleE', 'Coefficientofvariation', 'CVI', 'Correlationdimension', 
            'LFPowerLombScargle', 'aFdP', 'PSeo', 'PoincareSD1', 'sgridWGT', 'DFAAlpha2', 
            'LargestLyapunovexponent', 'pDpR', 'Hurstexponent', 'LF_HFratioLombScargle', 'histSI', 
            'DFAAlpha1', 'pD', 'PoincareSD2', 'fFdP', 'formF', 'SDLEalpha', 'SDLEmean',
            'VLFPowerLombScargle', 'KLPE', 'AsymI', 'Meanrate', 'CSI', 'IoV', 'gcount', 'shannEn', 'ARerr', 
            'PowerLawY_InterceptLombScargle', 'DFAAUC', 'QSE', 'Teo', 'Complexity']
    else:
        opts = []
        for e in lstEvents:
            opts = opts + list(e[waveform][minutes].keys())
        opts = list(set(opts))
    return sorted(opts)

###---LOAD FIGURES---###
''' Loop through the list of dicts, results, and get appropriate data for statistics graphs and boxplot (changepoint data)
    Return dictionary containing the data in dictionary form
    '''
def load_figure_source(metrics,results,minutes,waveform):
    source_dict={}
    source_dict["type"] = [evnt["event_type"] for evnt in results]
    source_dict["age"] = [evnt["age"] for evnt in results]
    source_dict["sex"] = [evnt["sex"] for evnt in results]
    source_dict["index"] = [evnt["_id"] for evnt in results]
    source_dict["vaso_type"] = [evnt["type"] for evnt in results]
    source_dict["admit_diagnosis"] = [ admitDiagnosisRetrieval(evnt["admit_diagnosis"]) for evnt in results ]
    source_dict["NonInvasiveMV"] = [evnt["NonInvasiveMV"] if "NonInvasiveMV" in list(evnt.keys()) else np.nan for evnt in results]
    source_dict["Dialysis"] = [evnt["Dialysis"] if "Dialysis" in list(evnt.keys()) else np.nan for evnt in results]
    source_dict["daySOFA"] = [evnt["daySOFA"] if "daySOFA" in list(evnt.keys()) else np.nan for evnt in results]
    source_dict["admitSOFA"] = [evnt["admitSOFA"] if "admitSOFA" in list(evnt.keys()) else np.nan for evnt in results]
    source_dict["APACHEII"] = [evnt["APACHEII"] if "APACHEII" in list(evnt.keys()) else np.nan for evnt in results]
    source_dict["InvasiveMV"] = [evnt["InvasiveMV"] if "InvasiveMV" in list(evnt.keys()) else np.nan for evnt in results]
    source_dict["vitalstatday28"] = [evnt["vitalstatday28"] if "vitalstatday28" in list(evnt.keys()) else np.nan for evnt in results]
    source_dict["label"] = [""]*len(results) #default no labels 
    source_dict["lactate_clearance"] = [evnt["lactate_patient_clearance"]["time_hours"] for evnt in results]
    source_dict["duration"] = [evnt["duration"] if "duration" in list(evnt.keys()) else np.nan for evnt in results]
    source_dict["other_pressors_on"] = [len(evnt["other_pressors_on"].keys()) if "other_pressors_on" in list(evnt.keys()) else np.nan for evnt in results]
    source_dict["pressor_restarted"] = [len(evnt["same_pressor_restarted"].keys()) if "same_pressor_restarted" in list(evnt.keys()) else np.nan for evnt in results]
    for metric in metrics:
        magnitudes = [ round(evnt[waveform][minutes][metric]["magnitude_change"],4) if metric in list(evnt[waveform][minutes].keys()) else np.nan for evnt in results ]
        locations = [ evnt[waveform][minutes][metric]["time_diff_change"] if metric in list(evnt[waveform][minutes].keys()) else np.nan for evnt in results ]
        pvalues = [ round(evnt[waveform][minutes][metric]["p-value"],4) if metric in list(evnt[waveform][minutes].keys()) and evnt[waveform][minutes][metric]["p-value"] is not None else np.nan for evnt in results ]
        source_dict[(metric+"_timediff")] = locations
        source_dict[(metric+"_mag")] = magnitudes
        source_dict[(metric+"_pvalue")] = pvalues
        source_dict["alphas"] = [0.9 if pvalue < 0.05 else 0.5 for pvalue in list(source_dict[(metric+"_pvalue")])] #for visualization of signficance
    return source_dict

###---FILTER FUNCTIONS---###
''' Loop through the list of dicts (lstEvents), get event filter_opt value that is in filter_values
    Return list of dicts (filtered) '''
def filterEventsLst(lstEvents,filter_opt,filter_values):
    #filter based on criteria
    filter_values = [x if x != "None" else None for x in filter_values] #replace "None" with None for seraching purposes
    filtered = [e for e in lstEvents if e[filter_opt] in filter_values]
    return filtered

''' From a bokeh buttonGroup get the label from the .active index
    Return list of labels '''
def getBtnGrpLabels(btngrp):
    return [ btngrp.labels[x] for x in btngrp.active ]

###---STATISTIC GRAPHS---###
def getStatsPercents(lst_val):
    from collections import Counter
    #send back stats for piechart - list of percentages
    total = len(lst_val)
    counts_dict = dict(Counter(lst_val))
    counts_dict = {k: v / total for k, v in counts_dict.items()}
    return counts_dict

def getProportions(lst_val):
    from collections import Counter
    counts_dict = dict(Counter(lst_val))
    counts_dict = {k: v for k, v in counts_dict.items()}
    return counts_dict

def getChi2Contingency(df,split_col,value_col):
    from scipy import stats
    lstoflsts = []
    unq = list(set(list(df[split_col])))
    keys = []
    for value in unq:
        sub_df = df.loc[df[split_col] == value]
        lst1 = list(sub_df[value_col])
        dctval = getProportions(lst1)
        keys.extend(list(dctval.keys()))
        lstoflsts.append(dctval)
    keys = list(set(keys))
    keys = [k for k in keys if str(k) != 'nan']
    #building the contingency table
    lstContingency = []
    for lst in lstoflsts:
        lstrow = []
        for k in keys:
            if k in list(lst.keys()):
                lstrow.append(int(lst[k]))
            else:
                lstrow.append(0)
        lstContingency.append(lstrow)
        lstrow = []
    chi2,pvalue,dof,expected = stats.chi2_contingency(lstContingency)
    return round(pvalue,4)

def getKruskalWallis(df,split_col,value_col):
    lstoflsts = []
    unq = list(set(list(df[split_col])))
    for value in unq:
        sub_df = df.loc[df[split_col] == value]
        lst1 = list(sub_df[value_col])
        lstoflsts.append(lst1)
    stat,pvalue = scipy.stats.kruskal(*lstoflsts,nan_policy="omit")
    return round(pvalue,4)

''' From lst_val (a list of raw values, categories), calculate binned data
    Return bin values, number in each bin and the step between bins'''
def getStatsHisto(lst_val):
    lst_val = lst_val.dropna()
    import numpy as np
    maxv = max(lst_val)
    minv = min(lst_val)
    if minv != maxv:
        bins, step = np.linspace(minv, maxv, (maxv-minv)/2, retstep=True)
        digitized = np.digitize(lst_val, bins)
        lst_val = np.array(lst_val)
        bin_counts = [len(lst_val[digitized == i]) for i in range(0, len(bins))]
    else: #only one selected, takes different formatting
        bins = [maxv-1,maxv,maxv+1]
        step = 0.5
        bin_counts = [0,1,0]
    return bins,bin_counts,step

''' Takes in a pandas df containing the data (columns) to graph and a list of the graph types - corresponding to each column
    Calculate the histo or categorical data based on type and df.column
    Return bokeh gridplot of statistics graphs and the sources from the graphs '''
def graphStats(df,clster_col):
    header = Div(text="""<h2> Selected Data </h2>""",height=30,style={"text-transform": "uppercase","text-align": "center","background-color": "lightgrey",'margin': '20px 0 0 0'})
    lst_graphs = [header]
    lst_sources = []
    source = ColumnDataSource(df)
    p1 = graphDotBoxplot(source,'lactate_clearance',clster_col, 'Lactate clearance', '', 'Lactate clearance (hrs)',tools_min=True)
    lst_graphs.append(p1)
    lst_sources.append(source)
    p6 = graphDotBoxplot(source, 'APACHEII', clster_col, 'APACHEII', '', 'APACHEII',tools_min=True)
    lst_graphs.append(p6)
    lst_sources.append(source)
    p9 = graphDotBoxplot(source, 'daySOFA', clster_col, 'Day SOFA', '', 'SOFA',tools_min=True)
    lst_graphs.append(p9)
    lst_sources.append(source)
    p11 = graphDotBoxplot(source,'pressor_restarted', clster_col, 'Pressor restarted within 12hrs', '', 'Count', tools_min=True)
    lst_graphs.append(p11)
    lst_sources.append(source)
    lstPercents, lstLabels, lstCate = getClusterStats(df[['sex',clster_col]],'sex',clster_col)
    p2,s2 = stackedBarChartMultiple(lstPercents, lstLabels, lstCate,title="Sex" )
    lst_graphs.append(p2)
    lst_sources.append(s2)
    if clster_col != "type":
        lstPercents, lstLabels, lstCate = getClusterStats(df[['type',clster_col]],'type',clster_col)
        p3,s3 = stackedBarChartMultiple(lstPercents, lstLabels, lstCate,title="Event type" )
        lst_graphs.append(p3)
        lst_sources.append(s3)
    lstPercents, lstLabels, lstCate = getClusterStats(df[['vaso_type',clster_col]],'vaso_type',clster_col)
    p4,s4 = stackedBarChartMultiple(lstPercents, lstLabels, lstCate,title="Vasopressor type"  )
    lst_graphs.append(p4)
    lst_sources.append(s4)
    lstPercents, lstLabels, lstCate = getClusterStats(df[['vitalstatday28',clster_col]],'vitalstatday28',clster_col)
    p5,s5 = stackedBarChartMultiple(lstPercents, lstLabels, lstCate,title="Vital Status"  )
    lst_graphs.append(p5)
    lst_sources.append(s5)
    header.width=p1.plot_width-10
    return [gridplot(lst_graphs,ncols=1,toolbar_location=None,merge_tools=False)],lst_sources

''' Takes in bokeh ColumnDataSource for the stats graph (original), type of analysis list, lst of new values
    Based on type, update source with new stats from list of new values '''
def updateStats(sourceorg,typeval,df_new,col,unq_sel,clster_col):
    colors = brewer["PuBuGn"][8]
    #print(col)
    if typeval == "cate":
        #['bottom', 'label_x', 'top', 'xval', 'label', 'index', 'color', 'label_y']
        lstPercents, lstLabels, lstCate = getClusterStats(df_new[[col,clster_col]],col,clster_col)
        num = len(lstCate)
        bottom = 0
        lst_dfs = []
        clusterorder = sorted(list(set(list(sourceorg.data["xval"]))))
        sourceorg.data = {x: [] for x in sourceorg.data}
        for x in range(len(lstCate)):
            percents = lstPercents[x]
            labels = lstLabels[x]
            xval = lstCate[x]
            try:
                pos = 0.25+ clusterorder.index(xval)
            except ValueError:
                pos = 0
            bottom = 0
            num = len(percents) #number of slices
            df = pd.DataFrame(columns=["bottom", "top", 'label_y', 'label','color','xval','label_x'],index=list(range(0,num)))
            for i in list(df.index):
                top = bottom+percents[i]
                label_y = bottom+(top-bottom)/2
                label_x = pos
                label_text = "("+str(round(percents[i]*100,1))+"%) "+labels[i]
                df.loc[i] = [bottom,top,label_y,label_text,colors[i],xval,label_x]
                bottom += percents[i]
            lst_dfs.append(df)
        final_df = pd.concat(lst_dfs)
        final_df.reset_index()
        sourceorg.stream({ "bottom": list(final_df["bottom"]),
        "label_x": list(final_df["label_x"]), "top": list(final_df["top"]),
        "xval": list(final_df["xval"]), "label": list(final_df["label"]),
        "index": list(final_df.index), "color": list(final_df["color"]),
        "label_y": list(final_df["label_y"]) })
    elif typeval == "histo":
        lstIndexesStats = list(sourceorg.data["index"])
        positions_stats = [ lstIndexesStats.index(x) for x in unq_sel ]
        sourceorg.selected = Selection(indices=positions_stats)

''' Get the percents, labels and categories for stackedBarChart visualization of categorical data. Return lists of
    percents, labels and categories. '''
def getClusterStats(df,col,clstr_col):
    unq_cate = list(set(list(df[clstr_col])))
    lstPercents = []
    lstLabels = []
    lstCate = unq_cate
    for cate in unq_cate:
        sub_df = df.loc[df[clstr_col]==cate]
        percents = getStatsPercents(sub_df[col])
        lstPercents.append(list(percents.values()))
        lstLabels.append(list(map(str,list(percents.keys()))))
    l1,l2,l3 = zip(*sorted(zip(lstCate, lstLabels,lstPercents))) #sort the three lists based on lstCate, return sorted lists
    return l3,l2,l1#lstPercents, lstLabels, lstCate

''' Get Outcome Panel graphs, stackedBarCharts and dotBoxPlots for the data. Return bokeh gridplot of the data. 
    Return grid plot of the outcome panel plots (column) '''
def getClusterCompStats(source,clstr_col):
    lstGraphs = []
    data = pd.DataFrame(source.data)
    pvalue1 = ""
    pvalue6 = ""
    pvalue2 = ""
    pvalue9 = ""
    pvalue3 = ""
    pvalue4 = ""
    pvalue5 = ""
    pvalue10 = ""
    pvalue11 = ""
    if len(set(list(data[clstr_col]))) > 1:
        pvalue1 = getKruskalWallis(data,clstr_col,'lactate_clearance')
        pvalue6 = getKruskalWallis(data,clstr_col,'APACHEII')
        pvalue9 = getKruskalWallis(data,clstr_col,'daySOFA')
        pvalue2 = getChi2Contingency(data,clstr_col,'sex')
        pvalue3 = getChi2Contingency(data,clstr_col,'type')
        pvalue4 = getChi2Contingency(data,clstr_col,'vaso_type')
        pvalue5 = getChi2Contingency(data,clstr_col,'vitalstatday28')
        pvalue10 = getChi2Contingency(data,clstr_col,'other_pressors_on')
        pvalue11 = getChi2Contingency(data,clstr_col,'pressor_restarted')
    p1 = graphDotBoxplot(source,'lactate_clearance',clstr_col, 'Lactate clearance (p:'+str(pvalue1)+')', '', 'Lactate clearance (hrs)',tools_min=True)    
    lstGraphs.append(p1) 
    p6 = graphDotBoxplot(source, 'APACHEII', clstr_col, 'APACHEII (p:'+str(pvalue6)+')','' , 'APACHEII',tools_min=True)
    lstGraphs.append(p6)
    p9 = graphDotBoxplot(source, 'daySOFA', clstr_col, 'Day SOFA (p:'+str(pvalue9)+')','' , 'SOFA',tools_min=True)
    lstGraphs.append(p9)
    p11 = graphDotBoxplot(source,'pressor_restarted', clstr_col, 'Pressors restarted within 12hrs (p:'+str(pvalue11)+')', '', 'Count', tools_min=True)
    lstGraphs.append(p11)
    lstPercents, lstLabels, lstCate = getClusterStats(data[['sex',clstr_col]],'sex',clstr_col)
    p2,s2 = stackedBarChartMultiple(lstPercents, lstLabels, lstCate,title="Sex (p:"+str(pvalue2)+')' )
    lstGraphs.append(p2)
    if clstr_col != "type":
        lstPercents, lstLabels, lstCate = getClusterStats(data[['type',clstr_col]],'type',clstr_col)
        p3,s3 = stackedBarChartMultiple(lstPercents, lstLabels, lstCate,title="Event type (p:"+str(pvalue3)+')' )
        lstGraphs.append(p3)
    lstPercents, lstLabels, lstCate = getClusterStats(data[['vaso_type',clstr_col]],'vaso_type',clstr_col)
    p4,s4 = stackedBarChartMultiple(lstPercents, lstLabels, lstCate,title="Vasopressor type (p:"+str(pvalue4)+')'  )
    lstGraphs.append(p4)
    lstPercents, lstLabels, lstCate = getClusterStats(data[['vitalstatday28',clstr_col]],'vitalstatday28',clstr_col)
    p5,s5 = stackedBarChartMultiple(lstPercents, lstLabels, lstCate,title="Vital Status Day 28 (p:"+str(pvalue5)+')'  )
    lstGraphs.append(p5)
    header = widgetbox(Div(text="""<h2> All Data </h2>""",width=p1.plot_width-10,height=30,
        style={"text-transform": "uppercase","text-align": "center","background-color": "lightgrey",'margin': '20px 0 0 0'}))
    lstGraphs.insert(0,header)
    return gridplot(lstGraphs,ncols=1,toolbar_location=None,merge_tools=False)

