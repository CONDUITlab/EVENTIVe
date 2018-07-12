# EVENTIVe

CONDUIT Lab

Creator: Victoria Tolls

Last Updated: July 2018

Code for EVENTIVe, an event-based interactive data visualization platform. EVENTIVe is aimed at studying micro-decisions in the ICU, however, the event-based functinality can branch to other fields of research.

## 



## Contents:

```callbacks.py``` - Contains the callback functions for widgets.

```cpt_hrv.py``` - Not used in EVENTIVe, but rather pre-processing into event-based data. Pettitt changepoint calculation.     Depends on r2py.
                            
```dotBoxplot.py``` - Modified boxplot Bokeh visualization.

```download.js``` - Javascript code to download a subset of data from a Bokeh ColumnDataSource.

```heatmap.py``` - Heatmap visualization with dendrogram hierarchical clustering.

```helpers.py``` - Helper functions used in main.py and callbacks.py

```indivStartCptPoint.py``` - Scatter plot with line of a time-series with percent change calculation, labelled startpoint and changepoint.

```kmeansClustering.py``` - K-Means clustering with colour coded scatter plot.

```main.py``` -  Main function. Loads webpage layout with widgets and reads in data source.

```pc_hrv.py``` - Percent change analysis for a time series.

```stackedBarChart.py``` - Stacked bar chart visualization.


## Dependencies:
* [pandas](https://pandas.pydata.org/pandas-docs/) v. 0.19.2
* [numpy](https://www.scipy.org/scipylib/download.html) v. 1.13.3
* [scipy](https://www.scipy.org/install.html)
* [bokeh](https://bokeh.pydata.org/en/latest/docs/reference.html)
* [sklearn](http://scikit-learn.org/stable/modules/clustering.html#clustering)

