# EVENTIVe-data-visualizations

CONDUIT Lab

Creator: Victoria Tolls

Last Updated: March 2018

A Sample of some of the visualizations I created for EVENTIVe. Dependent on scipy, sklearn, pandas, numpy and bokeh. 
Hierarchical Clustering and KMeans Clustering. Interactive visualizations using Bokeh. Main functions in each script are identified, each returns a bokeh plot.

## Contents:

hierarchical_heatmap_dendrogram_vis.py

kmeans_scatter_vis.py


## Examples:

### Heatmap

```
from bokeh.plotting import figure, output_file, show
from heatmap import heatmap

import pandas as pd
import numpy as np

df2 = pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)), columns=['a', 'b', 'c', 'd', 'e'], index=["s1","s2","s3","s4","s5"])

hmap,clusters = heatmap(df2,3,title="Sample Heatmap",plot_height=200)

show(hmap) 
```

### KMeans

```
from bokeh.plotting import figure, output_file, show
from kmeansClustering import kmeansClustering

import pandas as pd
import numpy as np

df2 = pd.DataFrame(np.random.randint(low=0, high=10, size=(10, 5)), columns=['a', 'b', 'c', 'd', 'e'], index=["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10"])

kmcluster,clusters = kmeansClustering(df2,3,"KMeans Sample",plot_width=400, plot_height=400)
show(kmcluster)
```


## Dependencies:
* [pandas](https://pandas.pydata.org/pandas-docs/)
* [numpy](https://www.scipy.org/scipylib/download.html)
* [scipy](https://www.scipy.org/install.html)
* [bokeh](https://bokeh.pydata.org/en/latest/docs/reference.html)
* [sklearn](http://scikit-learn.org/stable/modules/clustering.html#clustering)
