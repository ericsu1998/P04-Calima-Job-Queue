import sys
import math
import statistics
from datetime import timedelta
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np

sys.path.insert(0, '/pylon5/cc5fp8p/esu1/projects/p04/predictJET/torchNN')
from misc import load

#Job ID,Predicted Start Time,Actual Start Time,Eligible

def timeDeltaToHours(td):
    secondsInADay = float(3600*24)
    days = td.days + td.seconds / secondsInADay
    hoursPerDay = 24
    return days*hoursPerDay

def extractWaitTimes(times):
    fields = list(times.values())
    fields = list(filter(lambda x: None not in x, fields))
    fields = list(filter(lambda x: "Unknown" not in x, fields))
    predictedStart = [x[0] for x in fields]
    actualStart = [x[1] for x in fields]
    eligible = [x[2] for x in fields]
    predictedQWT = [timeDeltaToHours(predictedStart[i] - eligible[i])
                    for i in range(len(eligible))]
    actualQWT = [timeDeltaToHours(actualStart[i] - eligible[i])
                 for i in range(len(eligible))]
    return predictedQWT, actualQWT

def relativeError(predictedQWT, actualQWT):
    return [(predictedQWT[i] - actualQWT[i]) / actualQWT[i] * 100
             for i in range(len(actualQWT))]

def MSE(L):
    n = len(L)
    squaredL = [pow(x,2) for x in L]
    return sum(squaredL)/n

def printStats(errors):
    N = len(errors)
    print("N:", N)
    #nOverPredictions = len(list(filter(lambda x: x > 0, errors)))
    #nUnderPredictions = N - nOverPredictions
    #print("Number of Overpredictions: ", nOverPredictions)
    #print("Overprediction %: {:0.2f}".format(nOverPredictions/N))
    #print("Number of Underpredictions: ", nUnderPredictions)
    #print("Underprediction %: {:0.2f}".format(nUnderPredictions/N))
    print("Mean:", round(statistics.mean(errors), 2), "Hrs")
    print("Median:", round(statistics.median(errors), 2), "Hrs")
    print("Standard Deviation:", round(statistics.stdev(errors), 2), "Hrs")
    #print("MSE:", MSE(errors))
    #print("RMSE:", math.pow(MSE(errors), 0.5))

def goodPredictions(relativeErrors, threshold = 15):
    n = len(relativeErrors)
    goodPrediction = lambda error: abs(error) < threshold
    goodPredictions = list(filter(goodPrediction, relativeErrors))
    return len(goodPredictions) / float(n) * 100

def isOutlierFn(L):
    #Returns a function that return true if x is not an outlier,
    #and false if x is an outlier
    IQR = scipy.stats.iqr(L)
    L = np.array(L)
    Q1 = np.percentile(L, 25)
    leftCutoff = Q1 - 1.5*IQR
    Q3 = np.percentile(L, 75)
    rightCutoff = Q3 + 1.5*IQR
    print("Left outlier:",leftCutoff)
    print("Right outlier:", rightCutoff)
    isOutlier = lambda x: x < leftCutoff or rightCutoff < x
    return isOutlier

def removeOutliers(x, y):
    #Removes x's outliers from both x and y
    #Assumption: x and y are both the same size
    xIsOutlier = isOutlierFn(x)
    newX = []
    newY = []
    for i in range(len(x)):
        if not xIsOutlier(x[i]):
            newX.append(x[i])
            newY.append(y[i])
    return newX, newY

def graphErrors(x, y, xlabel="", ylabel=""):
    plt.plot(x, y, "ro")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == "__main__":
    times = load(sys.argv[1])
    predictedQWT, actualQWT = extractWaitTimes(times)
    relativeErrors = relativeError(predictedQWT, actualQWT)
    goodPredictionPercentage15 = goodPredictions(relativeErrors)
    print("Good prediction % when threshold = 15:", round(goodPredictionPercentage15, 2))
    goodPredictionPercentage10 = goodPredictions(relativeErrors, threshold=10)
    print("Good prediction % when threshold = 10:", round(goodPredictionPercentage10, 2))

    (relativeErrorsNoOutliers, actualQWTNoOutliers) = removeOutliers(relativeErrors, actualQWT)
    graphErrors(relativeErrorsNoOutliers, actualQWTNoOutliers,
                xlabel="Relative Error (%)", ylabel="Queue Wait Time (hrs)")
