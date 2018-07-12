############################################################
## CONDUIT Lab                                            ##
## January 2018                                           ##
## Creator: Victoria Tolls                                ##
## cpt_hrv.py                                             ##
## Calculate changepoint in a time series. Uses R pettitt ##
## test to calcualte changepoint. Pandas df taken as      ##
## shape is (nxm). Also return location of changepoint    ##
## and magnitude of the changepoint                       ##
############################################################

#--------------------------------------------------------------------------#
#----------------------------- CHANGEPOINT --------------------------------#
#--------------------------------------------------------------------------#

def calculateChangePoint(data):
    #import rpy2 to run R graphing 
    import rpy2
    #the R function for graphing
    string = """
        calculateCP <- function(data, time){
            #changepoint
            require(trend) 
            #require(ecp)          

            #intercepts and variable name vectors, empty until filled
            intercepts <- c()
            varnames <- c()
            indexes <- c()
            p_values <- c()
           
            #--create a vector, remove NaN
            d <- unlist(data,use.names=FALSE)
            d <- na.omit(d)
            #--pettitt test
            petest <- pettitt.test(d)
            #--save values
            p_values <- c(p_values, petest$p.value[1])
            #intercepts <- c(intercepts, time[petest$estimate])
            indexes <- c(indexes, as.integer(petest$estimate))
                               
            #create list for multiple return in R
            final = list(indexes, p_values)
            return(final)
        }
    """
    #load rpy2 and the string above as an R function
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
    calculatecpt = SignatureTranslatedAnonymousPackage(string, "calculatecpt")
    #if the dataframes are not empty, graph the plots
    if data != []:
        results = calculatecpt.calculateCP(data)
        return results
    else:
        return None
#calculate magnitude difference between the changepoint and the point just before
def calculate_magnitude(values, cpt, stime):
    if cpt is not None and len(values) > 1:
        return values[cpt] - values[stime]
    else:
        return None
#calculate the minute difference between the changepoint and the start of the event
def calculate_location(times, start, cpt):
    from datetime import datetime, timedelta
    if cpt is not None and len(times) > 1:
        #start = times[stime]
        changepoint = times[cpt]
        tchange = changepoint - start
        #negative value
        if tchange.total_seconds() < 0.0:
            final = (tchange.total_seconds()/60)
        else: 
            final = (tchange.total_seconds()/60)
        locationPC = str(final)
        return locationPC
    else:
        return None