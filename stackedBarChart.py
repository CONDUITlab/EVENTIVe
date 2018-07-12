############################################################
## CONDUIT Lab                                            ##
## February 2018                                          ##
## Creator: Victoria Tolls                                ##
## stackedBarChart.py                                     ##
## Create stacked bar chart of percents. Takes in list of ##
## percents and corresponding list of labels. Returns     ##
## bokeh figure.                                          ##
############################################################

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource,Legend,Label, LabelSet
from numpy import pi
from bokeh.palettes import d3,mpl,brewer
import pandas as pd
import math

''' FOR ONE CATEGORY ONLY. Create a stacked bar chart of percents. Pass in list of percents, list of matching labels. 
    Return bokeh plot. '''
def stackedBarChart(percents,labels,title=""):
    # a color for each pie piece
    colors = d3["Category20"][20]
    #get longest label (to format width of graph)
    str_width = len(max(labels, key=len))
    range_max = ((100+(str_width*5)) / 100)
    plot_tools = {'plot_height': 350,'plot_width': 100+(str_width*20), 'tools':['save']}
    #CREATE BOKEH FIGURE#
    p = figure(title=title,**plot_tools,x_range=(-0.30,range_max))
    bottom = 0
    num = len(percents) #number of slices
    #create pandas df with bottom, top, label y position, label text and color for each section of each bar
    df = pd.DataFrame(columns=["bottom", "top", 'label_y', 'label','color'],index=list(range(0,num)))
    #fill dataframe
    for i in list(df.index):
        top = bottom+percents[i]
        label_y = bottom+(top-bottom)/2
        label_text = "("+str(round(percents[i]*100,1))+"%) "+labels[i]
        df.loc[i] = [bottom,top,label_y,label_text,colors[i]]
        bottom += percents[i]
    #create bokeh columndatasource
    source = ColumnDataSource(df)
    #add bars
    p.vbar(x=0,bottom='bottom',top='top',width=0.5,alpha=0.7,color='color',source=source)
    #add text labels
    p.add_layout(LabelSet(x=0,y='label_y',text='label',text_font_size="8pt",source=source))
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.outline_line_width = 3
    p.toolbar.logo = None
    return p,source #return bokeh figure

''' FOR MULTIPLE CATEGORIES. Create a stacked bar chart of percents. Pass in list of lists [[percents1],[percents2]] and
    matching list of lists for labels [[labels1], [labels2]], list of categories [cate1, cate2]. Return bokeh plot. '''
def stackedBarChartMultiple(lstLstPercents,lstLstLabels, lstCate,title=""):
    colors = brewer["PuBuGn"][8]
    size = 200+(len(lstCate)*95) #set width based on the number of categories
    #CREATE BOKEH FIGURE
    plot_tools = {'plot_height': 250,'plot_width': size, 'tools':['save']}
    p = figure(title=title,**plot_tools,x_range=sorted(lstCate))
    lst_dfs = [] #empty list, for pandas df
    count = 0.25
    #loop through categories to create dfs for each set of percents and labels
    for x in range(len(lstCate)):
        percents = lstLstPercents[x]
        labels = lstLstLabels[x]
        xval = lstCate[x]
        bottom = 0
        num = len(percents) #number of slices
        #create pandas df with bottom, top, label y position, label text and color for each section of each bar
        df = pd.DataFrame(columns=["bottom", "top", 'label_y', 'label','color','xval','label_x'],index=list(range(0,num)))
        #fill dataframe
        for i in list(df.index):
            top = bottom+percents[i]
            label_y = bottom+(top-bottom)/2
            label_x = count
            label_text = "("+str(round(percents[i]*100,1))+"%) "+labels[i]
            df.loc[i] = [bottom,top,label_y,label_text,colors[i],xval,label_x]
            bottom += percents[i]
        lst_dfs.append(df)
        count = count+1
    #concatenate the dataframes
    final_df = pd.concat(lst_dfs)
    final_df.reset_index()
    #create bokeh columndatasource
    source = ColumnDataSource(final_df)
    #add bars
    p.vbar(x='xval',bottom='bottom',top='top',width=0.5,alpha=0.7,color='color',source=source)
    #add text labels
    p.add_layout(LabelSet(x='label_x',y='label_y',text='label',text_font_size="8pt",source=source)) #text_color="#647592"
    p.yaxis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.toolbar.logo = None
    return p,source #return bokeh figure