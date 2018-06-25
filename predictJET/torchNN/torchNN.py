import sys
import re
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocessing import getHeaderNames, preprocess, hasMissingValues, getMappings, toSparseTensor
from misc import TimeTracker, save, load
import math
import matplotlib.pyplot as plt

#Pytorch Neural Network for regression

def timeStr2Time(timeStr):
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

def trainNN(X, Y, nHidden = 1000, lRate = 1e-4, nEpochs=100):
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

        print(t, math.sqrt(loss.data[0]))

        model.zero_grad()

        loss.backward()

        for param in model.parameters():
            param.data -= lRate * param.grad.data

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

if __name__ == "__main__":
    logFile = sys.argv[1]
    timeTracker = TimeTracker()

    #"""
    #Get features/labels
    timeTracker.startBlock("Getting features/labels")

    same = lambda x: x
    timeElapsedStr2Time = [same, same, timeStr2Time]
    features, labels = preprocess(logFile, sep="|", header=True,
                                  featuresMapFns=timeElapsedStr2Time,
                                  labelMapFn=timeStr2Time)

    mappings = getMappings(features, nCat=2)

    #Save objects
    timeTracker.changeBlock("Saving data")
    save(features, "savedObjs/features_6_25.pickle")
    save(labels, "savedObjs/labels_6_25.pickle")
    save(mappings, "savedObjs/mappings_6_25.pickle")

    timeTracker.endBlock()
    #"""

    #"""
    #Load objects
    timeTracker.startBlock("Loading data")
    features = load("savedObjs/features_6_25.pickle")
    labels = load("savedObjs/labels_6_25.pickle")
    mappings = load("savedObjs/mappings_6_25.pickle")

    #Split to train/test
    timeTracker.changeBlock("Splitting to train/test")
    trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(
        features, labels, test_size=0.2)

    #Convert to Tensor
    timeTracker.changeBlock("Converting to Tensor")
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

    #Save Model and test data
    timeTracker.changeBlock("Saving model and test data")
    torch.save(testFeatures, "savedObjs/testFeatures_6_25.pickle")
    torch.save(testLabels, "savedObjs/testLabels_6_25.pickle")
    torch.save(testFeaturesTensor, "savedObjs/testFeaturesTensor_6_25.pickle")
    torch.save(testPredict, "savedObjs/testPredict_6_25.pickle")
    torch.save(model, "savedObjs/model_6_25.pickle")

    timeTracker.endBlock()
    #"""

    #"""
    #Load Model and test data
    timeTracker.startBlock("Loading model and test data")
    #model = torch.load("savedObjs/model_6_25.pickle")
    testFeatures = torch.load("savedObjs/testFeatures_6_25.pickle")
    #testFeaturesTensor = torch.load("savedObjs/testFeaturesTensor_6_25.pickle")
    testLabels = torch.load("savedObjs/testLabels_6_25.pickle")
    testPredict = torch.load("savedObjs/testPredict_6_25.pickle")

    #Output predictions
    timeTracker.changeBlock("Outputting predictions")
    outFile = "test.out"
    headerNames = getHeaderNames(logFile, sep="|")
    featureNames = headerNames[:-1]
    labelName = headerNames[-1]
    outputPredictions(outFile, featureNames, testFeatures, labelName, testLabels, testPredict)

    #Evaluate how good predictions are
    timeTracker.changeBlock("Outputting error")
    trainError = error(trainPredict, trainLabels)
    testError = error(testPredict, testLabels)
    errorFile = "error.out"
    outputError(errorFile, trainError, testError)

    timeTracker.endBlock()
    #"""
