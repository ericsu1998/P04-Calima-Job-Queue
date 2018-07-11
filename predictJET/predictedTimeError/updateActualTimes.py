import sys
from datetime import datetime, timedelta

sys.path.insert(0, '/pylon5/cc5fp8p/esu1/projects/p04/predictJET/torchNN')
from preprocessing import preprocess
from misc import save, load

def toDateTime(timeStr):
    if timeStr == "Unknown":
        return timeStr
    else:
        return datetime.strptime(timeStr, "%Y-%m-%dT%H:%M:%S")

def updateTimes(timesFile, times, actualTimes):
    updatedCount = 0
    for jid in times:
        if jid in actualTimes and times[jid][1] == None:
            if actualTimes[jid] != "Unknown":
                times[jid][1] = actualTimes[jid]
                predictionError = times[jid][0] - actualTimes[jid]
                times[jid][2] = predictionError

                print(times[jid])
                updatedCount += 1

    print("Number updated:", updatedCount)
    save(times, timesFile)
    return times

def outputTimes(outFile, times):
    with open(outFile, "w") as f:
        f.write("Job ID,Predicted Time,Actual Time,Time Difference\n")
        for jid in times:
            f.write(jid)
            f.write(",")
            for field in times[jid]:
                f.write(str(field))
                f.write(",")
            f.write("\n")

if __name__ == "__main__":
    logs = sys.argv[1]
    timesFile = sys.argv[2]
    outFile = "times.out"
    times = load(timesFile)

    actualTimes = preprocess(logs, featuresMapFns=[toDateTime],
                             header=True, asDict=True)

    for JobID in actualTimes:
        actualTimes[JobID] = actualTimes[JobID][0]

    newTimes = updateTimes(timesFile, times, actualTimes)
    outputTimes(outFile, newTimes)
