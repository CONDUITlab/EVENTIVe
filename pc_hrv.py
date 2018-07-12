############################################################
## CONDUIT Lab                                            ##
## Janurary 2018                                          ##
## Creator: Victoria Tolls                                ##
## pc_hrv.py                                              ##
## Percent Change analysis, based on paper by Dr. A.      ##
## Seely. Get baseline value from before specific time    ##
## calculate difference based on threshold, any change    ##
## above or below the threshold is labelled significant   ##
## everything else is labelled baseline.                  ##
############################################################

#-----------------------------------------------------------------------------------------#
import pandas as pd

#retrive the baseline from the df
def getBeforeDF(df_fs, t, bmin):
    b = df_fs.query('time < @t')
    b_f = b.query('time > -@bmin')
    return b_f
def getAverageBaselineDF(df):
    import numpy as np
    value = df.columns[1]
    #if the dataframe is empty, return np.nan
    if df.empty:
        return np.nan
    else:
        return (sum(df[value])/len(df))
#calculate the difference between the baseline and the new data, based on a threshold
#return dataframe of labelled points
def getLabeledPercentChangeDF(df, baseline, threshold):
    import numpy as np
    import pandas as pd
    nobs = df.shape[0]
    value = df.columns[1]
    pc = pd.DataFrame({value: range(nobs)})
    pc_v = pd.DataFrame({value: range(nobs)})
    df.index = range(nobs)
    #No baseline value, not present in data
    #return df of nan
    if baseline == np.nan:
        for k in range(nobs):
            pc[value][k] = 'baseline'
    else:
        #if above threshold or below -threshold mark as significant, otherwise do baseline
        for k in range(nobs):
            x = ((baseline - df[value][k])/baseline)*100
            x = float(round(x,10))
            if np.isinf(x):
                x = np.nan
            elif x >= threshold or x <= -threshold:
                pc[value][k] = 'significant'
            else:
                pc[value][k] = 'baseline'
            pc_v[value][k] = x
    return pc, pc_v