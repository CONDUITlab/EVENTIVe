from bokeh.plotting import figure, output_file, show
from kmeans_scatter_vis import kmeansClustering

import pandas as pd
import numpy as np

df2 = pd.DataFrame(np.random.randint(low=0, high=10, size=(10, 5)), columns=['a', 'b', 'c', 'd', 'e'], index=["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10"])

kmcluster,clusters = kmeansClustering(df2,3,"KMeans Sample",plot_width=400, plot_height=400)
show(kmcluster)
