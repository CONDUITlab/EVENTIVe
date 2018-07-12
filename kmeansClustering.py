############################################################
## CONDUIT Lab                                            ##
## January 2018                                           ##
## Creator: Victoria Tolls                                ##
## kmeansClustering.py                                    ##
## Code to create a scatter plot of clustering. nxm       ##
## pandas dataframe used. Kmeans run on entire dataframe  ##
## then PCA on entire dataframe. PC1 and PC2 used for     ##
## scatter plot graphing, color coded to kmeans clusters. ##
############################################################

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from sklearn.decomposition import PCA
from bokeh.models import ColumnDataSource, Label
from bokeh.models import HoverTool
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
    if n_clusters == 0:
        km, y_km = silhouetteAnalysis(X)
    else:
        km = KMeans(n_clusters=int(n_clusters)).fit(X)
        y_km = km.predict(X)
    #label PCA data with colors from kmeans
    df2d['color'] = list(map(str, y_km))
    df2d['index'] = list(df.index)
    source = ColumnDataSource(df2d)
    ##CUSTOM HOVER TOOL##
    hover = HoverTool(tooltips=[
        ("id", "@index"),
    ])
    ##GRAPH CLUSTER##
    plot_tools = {'tools':['reset','save', hover, "box_select"]}
    color=factor_cmap('color', palette=palette, factors=sorted(list(set(map(str, y_km)))))
    p = figure(plot_width=plot_width, plot_height=plot_height,x_axis_label=xlbl, y_axis_label=ylbl,title=title, **plot_tools)
    p.circle('PC1','PC2',source=source,fill_color=color,alpha=0.9,line_color=color,size=9)
    p.toolbar.logo = None
    return p,source, df2d[["index","color"]]  #return graph, and color coded clusters

''' Use silhouette analysis to determine value of k. Return kmeans model and cluster labels. '''
def silhouetteAnalysis(X):
    km_final = None
    y_km_final = [0]*len(X) #to send back one cluster if none are found to match critera
    sil_avg_final = 0
    for n_clusters in range(3,11):
        #calculate the kmeans clusters
        km = KMeans(n_clusters=int(n_clusters)).fit(X)
        cluster_labels = km.predict(X)
        #get the silhouette average score - to be used as threshold
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        silhouettesabovethreshold = []
        clustersizes = []
        #determine how many silhouette's are above the threshold
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            checkallabove = any(i >= silhouette_avg for i in ith_cluster_silhouette_values)
            clustersizes.append(len(ith_cluster_silhouette_values))
            silhouettesabovethreshold.append(checkallabove)
        countabove = silhouettesabovethreshold.count(True)
        size = min(clustersizes)
        if countabove >= n_clusters/2 and size > 1 and silhouette_avg > sil_avg_final:
            km_final = km
            y_km_final = cluster_labels
            sil_avg_final = silhouette_avg
    return km_final, y_km_final

