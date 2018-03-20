############################################################
## CONDUIT Lab                                            ##
## January 2018                                           ##
## Creator: Victoria Tolls                                ##
## hierarchical_heatmap_dendrogram_vis.py                 ##
## Code to create a bokeh heatmap from a pandas data      ##
## frame data. (nxm) shape, with n patients of m          ##
## observations. With dendrogram.                         ##
############################################################

import pandas as pd
import numpy as np

from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import transform, factor_cmap
from bokeh.models import HoverTool
from bokeh.layouts import gridplot
from bokeh.palettes import mpl,d3,brewer

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage


''' Graph the heatmap, function call to get dendrogram, label dataset with dendrogram clusters to be returned.
    Data should be a nxm pandas dataframe with n patients/events (rows) with m parameters (columns), clustering uses all columns for all patients/events.
    ***Dataframe index used for "id" of patients/events, make sure that df.index is set to the unique ids if applicable '''
##MAIN FUNCTION##
def hierarchical_heatmap_dendrogram_vis(data,n_clusters,title="Heatmap",plot_width=500,plot_height=1300): 
    #copy the original data frame
    datakmeans = data.copy()
    #to handle NaN values, we replace with mean, for kmeans
    datakmeans.fillna(datakmeans.mean(),inplace=True)
    #get dendrogram, orderlist (ids/indexes that are in the same order as the dendrogram) 
    # and custers (a dictionary of key (cluster) and value (list of ids/indexes in that cluster)
    p1,ordered_lst,clusters = getDendrogram(datakmeans,n_clusters,plot_height)
    #reorder the data based on the dendrogram results
    data = data.loc[ordered_lst]
    #reshape the pandas df columns = level_0, level_1, metric
    df = pd.DataFrame(data.stack(), columns=['metric']).reset_index()
    df.columns=["index", "value", "metric"]
    df.index = list([df['index']])
    source = ColumnDataSource(df)
    #add additional data to the figure source for use later
    colors = mpl["Viridis"][256]
    #create color mapper
    mapper = LinearColorMapper(palette=colors, low=df.metric.min(), high=df.metric.max())
    plot_tools = {'tools':['reset','save',"box_select"]}
    #create bokeh figure
    p = figure(plot_width=plot_width, plot_height=plot_height, **plot_tools, title=title,
           y_range=[str(x) for x in list(data.index)], x_range=list(data.columns),
           toolbar_location="above", x_axis_location="below")
    p.rect(y="index", x="value", width=1, height=1, source=source,
       line_color=None, fill_color=transform('metric', mapper))
    color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                     ticker=BasicTicker(desired_num_ticks=4),
                     formatter=PrintfTickFormatter(format="%f"))
    p.add_layout(color_bar, 'left')
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.major_label_text_font_size = "7pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 1.0
    p.yaxis.visible = False
    p.xaxis.visible = False   
    p.toolbar.logo = None
    return gridplot([[p,p1]],merge_tools=False),source,clusters #return plot and labelled clusters (dictionary) 

#-------------------------------------------------------------------------------------------------------#
''' Convert data to matrix, create linkage, call to get dendrogram, return call to graph dendrogram '''
def getDendrogram(data,n_clusters,plot_height):
    m = data.as_matrix()
    link = linkage(m,"ward")
    den = get_den_with_cluster_num(link,n_clusters)
    return graphDendrogram(data,den,plot_height)

''' Using the dendrogram and data, plot the dendrogram and label data with dendrogram clustering '''
def graphDendrogram(data,den,plot_height):
    ordered = list(data.index[den["leaves"]]) #ordered index values, based on index of original "data" dataframe
    #get the event ids
    ids = data.loc[ordered]
    ids = list(ids.index)
    #return dict of label: event ids for each cluster
    clusters = labelClusters(den,ids)

    #get min value from matrix - format as list first with all values then search for min
    min_val = np.min(den['icoord'])
    max_val = np.max(den['icoord'])

    #shift the palette colors to account for the highest point on the dendrogram which is not actually a cluster
    #make those lines grey by inserting grey at that point in the palette list
    palette = list(d3["Category10"][10])
    nonlabel_value = list(np.setdiff1d(list(set(den["color_list"])),list(clusters.keys())))
    if nonlabel_value != []:
        nonlabel_ind = sorted(list(set(den["color_list"]))).index(nonlabel_value[0])
        palette.insert(nonlabel_ind,'#494D50')

    color=factor_cmap('color_list', palette=palette, factors=sorted(list(set(den["color_list"]))))
    source = ColumnDataSource({k: den[k] for k in ("dcoord", "icoord", "color_list")})
    #create bokeh figure
    p = figure(plot_width=100, plot_height=plot_height,y_range=(min_val-min_val/2,max_val+min_val/2),toolbar_location="above",tools=['save'])
    p.multi_line(xs='dcoord',ys='icoord', line_width=2, line_color=color,source=source)
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.min_border_left = 0
    p.min_border_right = 0
    p.min_border_top = 2
    p.min_border_bottom = 2
    p.axis.major_label_standoff = 0
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.border_fill_color = 'white'
    p.toolbar.logo = None
    
    #return figure, ordered list of indexes and labelled clusters
    return p, ordered,clusters

''' Find dendrogram with number of clusters set, if not found return default or closest number of clusters '''
def get_den_with_cluster_num(link,n_clusters):
    threshold = max(link[:,2])
    proportion = 1.0
    den_n_clusters = 20
    thresholds = []
    clusters_num = []
    den = None
    while(den_n_clusters != n_clusters and proportion > 0): #after 20 iterations stop, not going to change
        thresholds.append(threshold)
        clusters_num.append(den_n_clusters)
        den = dendrogram(link,color_threshold=threshold)
        den_n_clusters = len(set(den['color_list']))
        proportion = proportion-0.05
        threshold = max(link[:,2])*proportion
    if proportion < 0: #if you can't find exact num clusters, use closest amount
        idx = clusters_num.index(min(clusters_num, key=lambda x:abs(x-n_clusters)))
        threshold = thresholds[idx]
        den = dendrogram(link,color_threshold=threshold)
    #return threshold
    return den
    
''' Fucntion to label the clusters, based on the dendorgram data and the list of data (event ids) '''
def labelClusters(den,lst_ids):
    from collections import defaultdict
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))

    cluster_classes = defaultdict(list)
    for c, l in cluster_idxs.items():
        i_l = [lst_ids[i] for i in l]
        cluster_classes[c] = i_l

    #return the labelled classes and ids
    return cluster_classes
