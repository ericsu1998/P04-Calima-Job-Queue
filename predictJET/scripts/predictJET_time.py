import sys
import time
import re
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def processData(logs):
    with open(logs, "r") as f:
        #Remove headers
        lines = f.readlines()[2:]
        #Remove leading and trailing whitespace
        lines = list(map(lambda x: x.strip().split(" "), lines))
        #Remove in-between spaces
        lines = list(map(lambda x: filter(None, x), lines))
    return lines

def cleanData(jobsData, numParams):
    timeReq = None
    cleanedData = []
    for jobData in jobsData:
        if (len(jobData) == numParams-1): #No timeReq
            cleanedData.append([timeReq] + jobData)
        elif (len(jobData) == numParams):
            timeReq = jobData[0]
            cleanedData.append(jobData)

    return cleanedData

def timeStr2Time(timeStr):
    #Input: Time, in "[DD-[HH:]]MM:SS]" format
    #Output: Time in hours
    timeStr = re.sub("\+", "00", timeStr)
    daysHMS = timeStr.split("-")
    if (len(daysHMS) == 2):
        days = int(daysHMS[0])
        HMS = daysHMS[1]
    else:
        days = 0
        HMS = daysHMS[0]
    (hours, minutes, seconds) = list(map(int, HMS.split(":")))
    hrsPerDay = 24
    minsPerHr = 60.0
    secsPerHr = 60.0*60.0
    timeHrs = days*hrsPerDay + hours + minutes/minsPerHr + seconds/secsPerHr
    return timeHrs

def formatData(jobsData):
    for i in range(len(jobsData)):
        (strTimeReq, strJET) = jobsData[i]
        timeReq = timeStr2Time(strTimeReq)
        JET = timeStr2Time(strJET)
        jobsData[i] = [timeReq, JET]
    return jobsData

def testModel(jobsData, model):
    model.fit(X, y)

    y_train_pred = predict(model, X_train)
    trainError = error(y_train, y_train_pred)
    print "Train Error:", trainError

    y_pred = predict(model, X_test)
    testError = error(y_test, y_pred)
    print "Test Error:", testError
    print "Score:", model.score(X_test, y_test)
    return model

def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def checkLength(jobsData, numParams):
    #Debugging method
    numMissingData = 0
    for i in range(len(jobsData)):
        jobData = jobsData[i]
        if (len(jobData) != numParams):
            numMissingData += 1
            print jobData, i+3
            return False
    return True

if __name__ == "__main__":
    logs = sys.argv[1]

    numParams = 2 #TimeLimit, Elapsed
    start = time.time()
    jobsData = processData(logs)
    jobsData = cleanData(jobsData, numParams)
    print(checkLength(jobsData, numParams))
    jobsData = np.array(formatData(jobsData))

    X = jobsData[:, :-1]
    y = jobsData[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    train_test = X_train, X_test, y_train, y_test

    print "Ridge Regression with alpha = 0.5"
    ridge = linear_model.Ridge(alpha = 0.5)
    fittedRidge = testModel(train_test, ridge)
    print "Ridge Regression with generalized CV to find alpha"
    ridgeCV = linear_model.RidgeCV(alphas = np.arange(0.1, 1.1, 0.1))
    fittedRidgeCV = testModel(train_test, ridgeCV)
    print "Alpha:", fittedRidgeCV.alpha_

    print "Lasso Regression with CV to find alpha"
    lassoCV = linear_model.LassoLarsCV(cv=20)
    fittedLassoCV = testModel(train_test, lassoCV)

    end = time.time()
    print "Time:", end - start

