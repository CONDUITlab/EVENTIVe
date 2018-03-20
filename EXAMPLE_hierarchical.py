############################################################
## CONDUIT Lab                                            ##
## March 2018                                             ##
## Creator: Victoria Tolls                                ##
############################################################

## Example of how to use hierarchical heatmap and dendrogram visualization ##

from bokeh.plotting import figure, output_file, show
from hierarchical_heatmap_dendrogram_vis import hierarchical

import pandas as pd
import numpy as np

df2 = pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)), columns=['a', 'b', 'c', 'd', 'e'], index=["s1","s2","s3","s4","s5"])
hmap,source,clusters = heatmap(df2,3,title="Sample Heatmap",plot_height=200)
show(hmap)
