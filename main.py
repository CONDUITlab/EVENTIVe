############################################################
## CONDUIT Lab                                            ##
## January 2018                                           ##
## Creator: Victoria Tolls                                ##
## main.py                                                ##
## This code constructs and builds the bokeh document,    ##
## it contains all functionality of the main document.    ##
############################################################

import pandas as pd
import numpy as np
import os
from os.path import dirname, join

from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS, Jitter, Div
from bokeh.models.widgets import DataTable, TableColumn, RadioButtonGroup, PreText, RangeSlider, CheckboxButtonGroup,RadioGroup
from bokeh.models.widgets import Button, Tabs, Panel, MultiSelect, Slider
from bokeh.models.callbacks import OpenURL
from bokeh.layouts import widgetbox, layout, GridSpec
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.events import ButtonClick

from helpers import *
from callbacks import *

import time

#############################################################################
## ------------------------ LOAD CURDOC() -------------------------------- ##
#############################################################################
def load_curdoc():
     ##-------------------------- DEFINE CALLBACKS -----------------------------##
    def onClickBtnBoxPlot():
        #try and capture exceptions to prevent crashing
        try:
            #need to have some event selected to start process
            if multiselEvents.value == []:
                raise IndexError
            if metricOptions.active == []:
                raise IndexError
            txt_processing.style = {"font-size": '1.2em', 'font-weight': 'bold', 'color': 'SteelBlue'}
            txt_processing.text = """<pre>Processing <i class="fa fa-spinner fa-spin" style="font-size:30px"></i></pre>"""
            #change textbox values, telling user what is happening
            minutes = rdbtn_time.labels[rdbtn_time.active]
            waveform = str(rdbtn_waveform.labels[rdbtn_waveform.active])
            lstNewGraphs = clickBtnBoxPlot(dicts,multiselEvents.value,getBtnGrpLabels(metricOptions),minutes,waveform,col_indiv)
            tab_graphs.child.children = lstNewGraphs
            txt_processing.text = ""
        #errors
        except ValueError:
            txt_processing.style = {"font-size": '1.2em', 'font-weight': 'bold', 'color': 'red'}
            txt_processing.text = "<pre>More than one event type needed.</pre>"
        except KeyError:
            txt_processing.style = {"font-size": '1.2em', 'font-weight': 'bold', 'color': 'red'}
            txt_processing.text = "<pre>Keyerror.</pre>"
        except IndexError:
            txt_processing.style = {"font-size": '1.2em', 'font-weight': 'bold', 'color': 'red'}
            txt_processing.text = "<pre>No events with that metric found.</pre>" 

    def onClickBtnHeatmap():
        try:    
            #need to have some event selected to start process
            if multiselEvents.value == []:
                raise IndexError
            if metricOptions.active == []:
                raise IndexError
            if len(metricOptions.active) > 1:
                raise ValueError
            #change textbox values, telling user what is happening
            txt_processing.style = {"font-size": '1.2em', 'font-weight': 'bold', 'color': 'SteelBlue'}
            txt_processing.text = """<pre>Processing <i class="fa fa-spinner fa-spin" style="font-size:30px"></i></pre>"""
            minutes = rdbtn_time.labels[rdbtn_time.active]
            waveform = str(rdbtn_waveform.labels[rdbtn_waveform.active])
            lstNewGraphs = clickBtnHeatmap(dicts,multiselEvents.value,getBtnGrpLabels(metricOptions),minutes,waveform,rngSldrKmeans.value,col_indiv)
            tab_heatmaps.child.children = lstNewGraphs
            txt_processing.text = ""
        except KeyError:
            txt_processing.style = {"font-size": '1.2em', 'font-weight': 'bold', 'color': 'red'}
            txt_processing.text = "<pre>Keyerror.</pre>"
        except ValueError:
            txt_processing.style = {"font-size": '1.2em', 'font-weight': 'bold', 'color': 'red'}
            txt_processing.text = "<pre>Select only one metric for Heatmap.</pre>"
        except IndexError:
            txt_processing.style = {"font-size": '1.2em', 'font-weight': 'bold', 'color': 'red'}
            txt_processing.text = "<pre>No events with that metric found.</pre>"   


    def onClickApplyFiltersCkbxs(active):
        filtered = filterEventsLst(dicts,"event_type",[ckbx_evnt_type.labels[x] for x in ckbx_evnt_type.active]) #filter based on eventtype
        filtered = filterEventsLst(filtered,"type",[ckbx_type.labels[x] for x in ckbx_type.active]) #filter based on type
        options,active = loadEventOptions(filtered)
        multiselEvents.options = options
        multiselEvents.value = active

    def onClickBtnKMeansClustering():
        try:
            #need to have some event selected to start process
            if multiselEvents.value == []:
                raise IndexError
            if metricOptions.active == []:
                raise IndexError
            #change textbox values, telling user what is happening
            txt_processing.style = {"font-size": '1.2em', 'font-weight': 'bold', 'color': 'SteelBlue'}
            txt_processing.text = """<pre>Processing <i class="fa fa-spinner fa-spin" style="font-size:30px"></i></pre>"""
            minutes = str(rdbtn_time.labels[rdbtn_time.active])
            waveform = str(rdbtn_waveform.labels[rdbtn_waveform.active])
            lstNewGraphs = clickBtnKMeansClustering(dicts,multiselEvents.value,getBtnGrpLabels(metricOptions),minutes,waveform,rdoGrpKmeans.active,rngSldrKmeans.value,col_indiv)
            tab_clustering.child.children = lstNewGraphs
            txt_processing.text = ""
        except KeyError:
            txt_processing.style = {"font-size": '1.2em', 'font-weight': 'bold', 'color': 'red'}
            txt_processing.text = "<pre>Keyerror.</pre>"
        except IndexError:
            txt_processing.style = {"font-size": '1.2em', 'font-weight': 'bold', 'color': 'red'}
            txt_processing.text = "<pre>No events with that metric found.</pre>" 
    
    def onClickWaveformOpt(attr,old,new):
        metricOptions.labels = loadMetrics(dicts,rdbtn_waveform.labels[rdbtn_waveform.active],rdbtn_time.labels[rdbtn_time.active])
        
    ##---------------------------- LOAD DATA ---------------------------------##
    dicts = loadJsonData()

    ##-------------------------- CREATE WIDGETS -------------------------------##
    ###---Options for clustering---###
    rdoGrpKmeans = RadioGroup(labels=["Time surrounding event, all metrics", "After event, all metrics", 
        "Before event, all metrics", "First time point after event\n(select multiple metrics)"], 
        active=0,height=125)
    rdoGrpHeatmap = RadioGroup(labels=["Time surrounding event"], active=0, height=125)
    rngSldrKmeans =  Slider(start=0, end=10, value=0, step=1, title="N clusters",width=400,height=40)

    ###---Buttons---###
    btnBoxPlot = Button(button_type="warning", label="Time series Boxplot", sizing_mode="fixed",width=40)
    btnHeatmap = Button(button_type="warning", label="Hierarchical Clustering", sizing_mode="fixed",width=10)
    btnKMeansClustering = Button(button_type="warning", label="KMeans Clustering", sizing_mode="fixed", width=20)

    ###---Filtering options & widgets---###
    txt_type = PreText(text="Vasopressor Type:",width=200,height=10)
    ckbx_type = CheckboxButtonGroup(labels=["Norepipherine","Vasopressin", "Epinephrine", "Other", "Dobutamine","None"], 
        active=[0,1,2,3,4,5],button_type="primary")
    #event_type {P_ON, P_OFF, P_NONE, P_STABLE}
    txt_evnt_type = PreText(text="Event Type:",width=100,height=10)
    ckbx_evnt_type = CheckboxButtonGroup(labels=["P_ON", "P_OFF", "P_STABLE", "P_NONE"], 
        active=[0,1,2,3],button_type="primary")
    #time {20,30}
    txt_time = PreText(text="Minutes surrounding event start:",width=400,height=10)
    rdbtn_time = RadioButtonGroup(labels=["20", "30"], active=0,button_type="primary")
    #MAP or HRV
    rdbtn_waveform = RadioButtonGroup(labels=["HRVMetrics", "VitalSigns"], active=0,button_type="primary",height=45)

    ###---Options for selecting the events to graph---###
    options,active = loadEventOptions(dicts)
    multiselEvents = MultiSelect(title="Events:", value=active, options=options,
        sizing_mode="scale_width", width=200,size=17)
    txt_metrics = PreText(text="HRV Metrics:",width=100,height=10)
    metricOptions = CheckboxButtonGroup(labels=loadMetrics(dicts,rdbtn_waveform.labels[rdbtn_waveform.active],rdbtn_time.labels[rdbtn_time.active]),#loadMetricsMIN(),
        active=[0],button_type="default")

    ###---Figure groupings---###
    col_figures = column(name="figures", sizing_mode='scale_width')
    col_indiv = column(sizing_mode='scale_width')
    col_heatmaps = row(name="heatmaps",sizing_mode="scale_width")
    col_clustering = column(name="clustering",sizing_mode="scale_width")

    ###---HTML code---##
    banner = Div(text=open(join(dirname(__file__), "static/banner.html")).read(),width=800,sizing_mode="fixed", style={'background-color': 'white'})
    header_analysis = Div(text="""<h2> Analysis Options </h2>""",width=150,height=20,style={"text-transform": "uppercase"})
    header_waveform = Div(text="""<h2> Waveforms </h2>""",width=150,height=20,style={"text-transform": "uppercase"})
    header_select_events = Div(text="""<h2> Filter Events </h2>""",width=300,height=20,style={"text-transform": "uppercase"})
    header_select_metrics = Div(text="""<h2> Select Metrics </h2>""",width=300,height=20,style={"text-transform": "uppercase"})
    header_datavis = Div(text="""<h2> Data Visualization </h2>""",width=200,height=20,style={"text-transform": "uppercase"})
    header_help_info = Div(text="""<h2> Help & Information </h2>""",width=300,height=20,style={"text-transform": "uppercase"})
    txt_processing = Div(text="",width=500,height=10)

    infoLink = Div(text=open(join(dirname(__file__), "static/helpButton.html")).read(),width=100)
    metricsExplnLink = Div(text=open(join(dirname(__file__), "static/hrvMetricExplainButton.html")).read(),width=175)

    ##------------------------ ASSIGN WIDGET CALLBACKS ------------------------##
    btnBoxPlot.on_click(onClickBtnBoxPlot)
    btnHeatmap.on_click(onClickBtnHeatmap)
    btnKMeansClustering.on_click(onClickBtnKMeansClustering)

    ckbx_evnt_type.on_click(onClickApplyFiltersCkbxs)
    ckbx_type.on_click(onClickApplyFiltersCkbxs)

    rdbtn_waveform.on_change('active',onClickWaveformOpt)

    ##----------------------- SET UP LAYOUT AND LOAD --------------------------##
    wdgbx_banner = widgetbox(banner)

    wdgbx_hrvmetrics = widgetbox(metricOptions,sizing_mode="scale_height",width=450,height=900,css_classes=["tabselect"]) #300

    layout_select = column(
        widgetbox(header_waveform,width=450,css_classes=["tabselect"]),  #350=500, 300=450, 150=250
        widgetbox(rdbtn_waveform,width=450,css_classes=["tabselect"]),
        widgetbox(Div(text="""<hr>""",sizing_mode="fixed",width=425),width=450,css_classes=["tabselect"]),
        widgetbox(header_analysis,width=450,css_classes=["tabselect"]),
        widgetbox(btnBoxPlot,width=450,css_classes=["tabselect"]),
        widgetbox(rngSldrKmeans,width=450,css_classes=["tabselect"]),
        row(widgetbox([btnKMeansClustering,rdoGrpKmeans],width=225,css_classes=["tabselect"]),widgetbox([btnHeatmap,rdoGrpHeatmap],width=225,css_classes=["tabselect"])),
        widgetbox(Div(text="""<hr>""",sizing_mode="fixed",width=425),width=450,css_classes=["tabselect"]),
        widgetbox(header_select_metrics,width=450,css_classes=["tabselect"]),
        wdgbx_hrvmetrics,
        widgetbox(Div(text="""<hr>""",sizing_mode="scale_width",width=425), width=450,css_classes=["tabselect"]),
        widgetbox(txt_time, rdbtn_time,width=450,css_classes=["tabselect"]),
        widgetbox(txt_evnt_type,ckbx_evnt_type,width=450,css_classes=["tabselect"]), 
        widgetbox(txt_type,ckbx_type,width=450,css_classes=["tabselect"]),
        widgetbox(Div(text="""<hr>""",sizing_mode="scale_width",width=425),width=450,css_classes=["tabselect"]),
        widgetbox(header_help_info,width=450,css_classes=["tabselect"]),
        row(widgetbox(infoLink,width=225,css_classes=["tabselect"]),widgetbox(metricsExplnLink,width=225,css_classes=["tabselect"])),
    )

    pan_options = Panel(child = layout_select, sizing_mode = "fixed", title="Option Panel",width=450,css_classes=["tabselect"])
    tab_select = Tabs(tabs=[pan_options],sizing_mode="fixed",width=450,css_classes=["tabselect_all"],height=1200)

    tab_graphs = Panel(child=col_figures, title="Time series Boxplot", sizing_mode="scale_width")
    tab_indiv_plots = Panel(child=col_indiv, title="Individual Plots", sizing_mode="scale_width")
    tab_heatmaps = Panel(child=col_heatmaps,title="Hierarchical Clustering",sizing_mode="scale_width")
    tab_clustering = Panel(child=col_clustering,title="KMeans Clustering",sizing_mode="scale_width")

    tabs= Tabs(tabs=[tab_graphs,tab_clustering,tab_heatmaps,tab_indiv_plots],sizing_mode="scale_width",width=1300)
    
    curdoc().title = "EVENTIVe"

    curdoc().add_root(column(wdgbx_banner, row(tab_select, column(row(header_datavis,txt_processing),tabs))))

#############################################################################
#---------------------------- MAIN FUNCTION --------------------------------#
#############################################################################
def main():
    load_curdoc()

#############################################################################
##----------------------------RUN PROGRAM ----------------------------------#
#############################################################################
main()