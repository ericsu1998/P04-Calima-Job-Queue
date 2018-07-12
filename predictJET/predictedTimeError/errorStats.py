import sys
import statistics
from datetime import timedelta
import matplotlib.pyplot as plt

sys.path.insert(0, '/pylon5/cc5fp8p/esu1/projects/p04/predictJET/torchNN')
from misc import load

def timeDeltaToHours(td):
    secondsInADay = float(3600*24)
    days = td.days + td.seconds / secondsInADay
    hoursPerDay = 24
    return days*hoursPerDay

def extractErrors(times):
    fields = times.values()
    errors = [x[2] for x in fields]
    errors = list(filter(None, errors))
    errors = list(map(timeDeltaToHours, errors))
    return errors

def printStats(errors):
    N = len(errors)
    print("N:", N)
    nOverPredictions = len(list(filter(lambda x: x > 0, errors)))
    nUnderPredictions = N - nOverPredictions
    print("Number of Overpredictions: ", nOverPredictions)
    print("Overprediction %: {:0.2f}".format(nOverPredictions/N))
    print("Number of Underpredictions: ", nUnderPredictions)
    print("Underprediction %: {:0.2f}".format(nUnderPredictions/N))
    print("Mean:", round(statistics.mean(errors), 2), "Hrs")
    print("Median:", round(statistics.median(errors), 2), "Hrs")
    print("Standard Deviation:", round(statistics.stdev(errors), 2), "Hrs")

def graphErrors(errors):
    plt.hist(errors)
    plt.xlabel("Error(hrs)")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    times = load(sys.argv[1])
    errors = extractErrors(times)
    printStats(errors)
    graphErrors(errors)
