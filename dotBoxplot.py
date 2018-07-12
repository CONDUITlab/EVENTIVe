#############################################################
## CONDUIT Lab                                             ##
## January 2018                                            ##
## Creator: Victoria Tolls                                 ##
## dotHistogram.py                                         ##
## Code to create a bokeh jitered (scatter) bar chart      ##
## from a pandas data frame data or bokeh columndatasource ##
# (nxm) shape, with n patients of m observations.          ##
#############################################################

from bokeh.models import ColumnDataSource, Span
from bokeh.plotting import figure
from bokeh.palettes import d3,mpl,brewer
from bokeh.transform import jitter, factor_cmap
from bokeh.models import HoverTool
import math
import pandas as pd

''' Dot Boxplot, a boxplot (upper, lower quartiles and median in a box) with the "jitter" points visible allowing for 
    selection of data. Takes in either a pandas df or bokeh ColumnDataSource containing data for graphing, along with
    graphing parameters. ** Xvalue needs to be categorical and Yvalue continuous** Returns bokeh plot.'''
## MAIN FUNCTION ##
def graphDotBoxplot(source, yvalue, xvalue, title="",x_label="", y_label="", tools_min=False):
    #pandas df passed in - create ColumnDataSource
    if isinstance(source,pd.DataFrame):
        df = source
        source = ColumnDataSource(data=df)
    #ColumnDataSource passed in - create pandas df from data
    elif isinstance(source,ColumnDataSource):
        df = pd.DataFrame(source.data)
    else:
        raise ValueError("Function takes either pandas dataframe or bokeh ColumnDataSource.") 
    palette = d3["Category10"][10]
    #print(df)
    #hover tool
    hover = HoverTool(tooltips=[
        ("id", "@index"),
    ])
    #minimum tools, only save button
    if tools_min == True:
        lstCate = list(set(list(df[xvalue])))
        size = 200+(len(lstCate)*95)
        plot_tools = {'plot_height': 250,'plot_width': size, 'tools':['save']}
    #more tools, allowing for more interaction with the data
    else:
        plot_tools = {'plot_height': 350,'plot_width': 450, 'tools':['save','reset','box_select','lasso_select','tap',hover],'active_drag':"box_select"}#, 'tap']}
    #CREATE BOKEH FIGURE
    x_range_sorted = sorted(list(df[xvalue].unique()))
    p1 = figure(name=yvalue, title=title,**plot_tools, x_range=x_range_sorted,x_axis_label=x_label, y_axis_label=y_label)
    x_values = jitter(xvalue, width=0.6, range=p1.x_range) #jitter the values to distribute them in the column
    color=factor_cmap(xvalue, palette=palette, factors=x_range_sorted)
    if "mag" in yvalue or "time" in yvalue:
        p1.circle(x=x_values, y=yvalue, source=source, alpha="alphas", size=9, fill_color=color, line_color=color)
    else:
        p1.square_x(x=x_values, y=yvalue, source=source, alpha=0.7, size=9, fill_color=None,line_color=color)
    p1.xgrid.grid_line_color = None
    p1.axis.major_label_text_font_size = "9pt"
    p1.toolbar.logo = None
    xind = 0
    #Add median, upper and lower quartile to create "box"
    for val in x_range_sorted:
        q2 = getMedian(df,xvalue,val,yvalue)
        q1,q3 = getQuartiles(df,xvalue,val,yvalue)
        #add median to each column
        if math.isnan(q2) == False:
            p1.segment(xind,q2,xind+1,q2,line_color='blue', line_width=2,alpha=0.5)
        #add upper and lower quartiles and lines to make it more box like
        if math.isnan(q1) == False and math.isnan(q3) == False: 
            p1.rect(xind+0.5,q1+((q3-q1)/2),1,q3-q1,line_color='black', line_width=2,alpha=0.5,fill_color=None)
        xind = xind+1 #shift x index to move over columns
    return p1

##---------------------------------------------------------------------------------------------------------##
''' Take in a pandas df, column name, value (category within the column), cal_value (column to calculate median for).
    Return quartile value. '''
def getMedian(df,column_name, value, cal_value):
    vals = df.loc[df[column_name] == value] #get sub df according to column name and value(category)
    q2 = vals[cal_value].quantile(q=0.50) # calculate median of cal_value
    return q2

''' Take in a pandas df, column name, value (category within the column), cal_value (column to calculate median for).
    Return upper and lower values. '''
def getQuartiles(df,column_name, value, cal_value):
    vals = df.loc[df[column_name] == value]
    q1 = vals[cal_value].quantile(q=0.25)
    q3 = vals[cal_value].quantile(q=0.75)
    return q1,q3