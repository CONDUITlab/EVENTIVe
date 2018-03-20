############################################################
## CONDUIT Lab                                            ##
## January 2018                                           ##
## Creator: Victoria Tolls                                ##
## kmeans_scatter_vis.py                                  ##
## Code to create a scatter plot of clustering. nxm       ##
## pandas dataframe used. Kmeans run on entire dataframe  ##
## then PCA on entire dataframe. PC1 and PC2 used for     ##
## scatter plot graphing, color coded to kmeans clusters. ##
############################################################

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from bokeh.models import ColumnDataSource, Label, HoverTool
from bokeh.plotting import figure
from bokeh.palettes import d3
from bokeh.transform import factor_cmap
import pandas as pd

''' Using pandas nxm dataframe (n events/patients, m parameters (columns)), number of clusters used to run
    Kmeans clustering on the pandas df, PCA on the pandas df. Bokeh graph of PC1 and PC2 with labelled 
    clusters from Kmeans. Returns bokeh figure.
'''
##MAIN FUNCTION ##
def kmeansClustering(df,n_clusters,title,plot_width=800, plot_height=600):
    #to handle NaN values, we replace with mean
    df = df.fillna(df.mean(),inplace=True)
    matrixData = df.as_matrix()
    palette = d3["Category10"][10]#list(reversed(d3["Category10"][10]))
    ##PCA ANALYSIS on original data##
    pca = PCA()
    pca.fit(matrixData)
    #transform to 2d array
    matrixData_2d = pca.transform(matrixData)
    #number of components
    num_comp = len(matrixData_2d[0])
    #pandas df of the 2d data
    df2d = pd.DataFrame(matrixData_2d)
    df2d.index = df.index
    df2d.columns = ['PC'+str(x) for x in range(1,num_comp+1)]
    xlbl = "PC1"
    ylbl = "PC2"
    ##KMEANS on orginal data##
    X = matrixData
    km = KMeans(n_clusters=int(n_clusters)).fit(X)
    y_km = km.predict(X)
    #label PCA data with colors from kmeans
    df2d['color'] = list(map(str, y_km))
    df2d['index'] = list(df.index)
    source = ColumnDataSource(df2d)
    ##GRAPH CLUSTER##
    plot_tools = {'tools':['reset','save', "box_select"]}
    color=factor_cmap('color', palette=palette, factors=sorted(list(set(map(str, y_km)))))
    p = figure(plot_width=plot_width, plot_height=plot_height,x_axis_label=xlbl, y_axis_label=ylbl,title=title, **plot_tools)
    p.square('PC1','PC2',source=source,fill_color=color,line_color=color,size=9)
    p.toolbar.logo = None
    return p,df2d[["index","color"]]  #return graph, and color coded clusters
