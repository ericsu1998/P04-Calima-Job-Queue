import sys
import re
import numpy as np
import torch
from scipy.stats import pearsonr
from datetime import datetime, timedelta
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocessing import getHeaderNames, preprocess, hasMissingValues, getMappings, toSparseTensor
from misc import TimeTracker, save, load
import math
import matplotlib.pyplot as plt

#Pytorch Neural Network for regression

def timeStr2Mins(timeStr):
    #Input: Time, in "[DD-[HH:]]MM:SS]" format
    #Output: Time in mins
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
    timeMins = timeHrs * 60
    return timeMins

def timeDeltaToMins(td):
    secondsInADay = float(3600*24)
    days = td.days + td.seconds / secondsInADay
    hoursPerDay = 24
    minsPerHour = 60
    return days*hoursPerDay*minsPerHour

def graphVars(features, labels):
    #Features has UID, timeLimit, nNodes
    #Labels is queueWaitTime
    queueWaitTime = labels
    timeLimit = [job[1] for job in features]
    print("Correlation coefficient for timeLimit vs queueWaitTime:",
          pearsonr(timeLimit, queueWaitTime)[0])
    #Plot timeLimit
    plt.plot(timeLimit, queueWaitTime, 'ro')
    plt.xlabel("Time Requested(min)")
    plt.ylabel("Queue Wait Time(min)")
    plt.show()

def graphErrors(error):
    plt.plot(error)
    plt.show()

def trainNN(X, Y, nHidden = 10, lRate = 1e-4, nEpochs=10, graphError=False):
    errors = []
    #Train NN on data using Adam
    #N is batch size: D_in is input dimension,
    #H is hidden dimension, D_out is output dimension
    dims = list(X.size())
    N, D_in = dims[0], dims[1]
    print ("N:", N, "D_in", D_in)
    H, D_out = nHidden, 1
    print("H:", H, "D_out", D_out)

    X = Variable(X)
    Y = Variable(Y, requires_grad=False)

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )
    lossFn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lRate)
    for t in range(nEpochs):
        yPred = model(X)

        loss = lossFn(yPred, Y)

        error = math.sqrt(loss.data[0])
        errors.append(error)
        print(t, error)

        model.zero_grad()

        loss.backward()

        for param in model.parameters():
            param.data -= lRate * param.grad.data

    if (graphError): graphErrors(errors)
    return model

def predict(model, testFeatures):
    testFeatures = Variable(testFeatures, requires_grad=False)
    predictions = model(testFeatures).data.numpy().flatten().tolist()
    return predictions

def outputPredictions(outFile, featureNames, features, labelName, labels, predictions):
    featureNameLengths = [len(featureName) for featureName in featureNames]
    actualLNLen = len(labelName) + len("(actual)")
    predictedLNLen = len(labelName) + len("(predicted)")

    with open(outFile, "w") as f:
        for featureName in featureNames:
            f.write("{}|".format(featureName))
        f.write("{}(actual)|{}(predicted)|Error\n".format(labelName, labelName))

        for i in range(len(predictions)):
            for feature in features[i]:
                f.write("{}|".format(feature))
            f.write("{:.2f}|{:.2f}|{:.2f}\n".format(labels[i], predictions[i], predictions[i]-labels[i]))

def error(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    RMSE = math.sqrt(mean_squared_error(predictions, labels))
    return RMSE

def outputError(errorFile, trainError, testError):
    with open(errorFile, "w") as f:
        f.write("Test error(RMSE): {:.2f}\n".format(trainError))
        f.write("Test error(RMSE): {:.2f}\n".format(testError))

def toDateTime(timeStr):
    if timeStr == "Unknown":
        return timeStr
    else:
        return datetime.strptime(timeStr, "%Y-%m-%dT%H:%M:%S")

if __name__ == "__main__":
    logFile = sys.argv[1]
    timeTracker = TimeTracker()

    #Get features/labels
    timeTracker.startBlock("Getting features/labels")

    same = lambda x: x
    featuresMapFns = [same, timeStr2Mins, int, toDateTime]
    skipFeatures = [None, None, None, "Unknown"]
    skipLabel = "Unknown"
    features, labels = preprocess(logFile, sep=",", header=True,
                                  featuresMapFns=featuresMapFns,
                                  labelMapFn=toDateTime,
                                  skipFeatures=skipFeatures,
                                  skipLabel=skipLabel)

    eligible = [x[-1] for x in features]
    features = [x[:-1] for x in features]
    labels = [timeDeltaToMins(labels[i] - eligible[i])
              for i in range(len(labels))]

    #Graph variables against labels (queueWaitTime)
    graphVars(features, labels)

    """
    #Split to train/test
    timeTracker.changeBlock("Splitting to train/test")
    trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(
        features, labels, test_size=0.2)

    #Convert to Tensor
    timeTracker.changeBlock("Converting to Tensor")

    mappings = getMappings(features, nCat=1)
    trainFeaturesTensor = toSparseTensor(trainFeatures, mappings).to_dense()
    testFeaturesTensor = toSparseTensor(testFeatures, mappings).to_dense()

    trainLabelsTensor = torch.Tensor(trainLabels)

    #Train model on trainingData
    timeTracker.changeBlock("Training model")
    model = trainNN(trainFeaturesTensor, trainLabelsTensor)
    trainPredict = predict(model, trainFeaturesTensor)

    #Test model on testing data
    timeTracker.changeBlock("Predicting on test data")
    testPredict = predict(model, testFeaturesTensor)
    #Output predictions
    timeTracker.changeBlock("Outputting predictions")
    outFile = "test.out"
    featureNames = ["UID", "TimeLimit", "NNodes"]
    labelName = "QWT(Mins)"
    outputPredictions(outFile, featureNames, testFeatures, labelName, testLabels, testPredict)

    #Evaluate how good predictions are
    timeTracker.changeBlock("Outputting error")
    trainError = error(trainPredict, trainLabels)
    testError = error(testPredict, testLabels)
    errorFile = "error.out"
    outputError(errorFile, trainError, testError)

    timeTracker.endBlock()
    """
