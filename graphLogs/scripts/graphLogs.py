import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

#TODO: Header file

def processData(fileName):
    with open(fileName, "r") as f:
        #Remove headers
        lines = f.readlines()[2:]
        #Remove leading and trailing whitespace
        lines = list(map(lambda x: x.strip().split(" "), lines))
        #Remove in-between spaces
        lines = list(map(lambda x: filter(None, x), lines))
    return lines

def parseLogs(fileName):
    #Input: fileName with slurm logs containing partition, submit, and start
    #Output: (days, partitions, queueWaitTimes) where
    #   days = list of days that the job was ran in
    #          YYYY-MM-DD format
    #   partitions = list of partitions that jobs were queued in
    #                (RM, RM-shared, GPU, LM, etc.)
    #   queueWaitTimes: list of how long jobs waited in queue in
    #                   timedelta format

    partitions = []
    days = []
    queueWaitTimes = []

    jobs = processData(fileName)
    prevPartition = None
    for i in range(len(jobs)):
        if (len(jobs[i]) != 3):
            partition = prevPartition
            start = jobs[i][0]
            submit = jobs[i][0]
        else:
            partition = jobs[i][0]
            prevPartition = partition
            start = jobs[i][1]
            submit = jobs[i][2]

        if (start == "Unknown"): continue
        day = submit.split("T")[0]
        day = datetime.strptime(day, "%Y-%m-%d")
        startF = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
        submitF = datetime.strptime(submit, "%Y-%m-%dT%H:%M:%S")
        queueWaitTime = startF - submitF
        queueWaitTimes.append(queueWaitTime)
        days.append(day)
        partitions.append(partition)

    return (days, partitions, queueWaitTimes)

def timeDeltaToTime(td):
    secondsInADay = float(3600*24)
    return td.days + td.seconds / secondsInADay

def mapPartDict(partDict):
    for partition, td in partDict.iteritems():
        partDict[partition] = timeDeltaToTime(td)
    return partDict

def mapTDToTime(tdDict):
    for day, partition in tdDict.iteritems():
        tdDict[day] = mapPartDict(partition)
    return tdDict

def splitDayPartition(days, partitions, queueWaitTimes):
    #TODO (?): include time duration (e.g. day, week, etc.)
    #Inputs:
    #   days = list of days that the job was ran in
    #          YYYY-MM-DD format
    #   partitions = list of partitions that jobs were queued in
    #                (RM, RM-shared, GPU, LM, etc.)
    #   queueWaitTimes: list of how long jobs waited in queue in
    #                   timedelta format
    #   All 3 lists should have the same length

    #Outputs:
    #   Dictionary where keys are days,
    #   and values are another dictionary where
    #   keys are partitions and values are cumulative weight times
    #   The format is illustrated below:
    #   {day1 : {partition1 : waitTime11, partition2: waitTime12}
    #    day2 : {partition1 : waitTime21, partition2 : waitTime22}
    #   }

    timeDict = dict()
    nJobsDict = dict()
    for i in range(len(days)):
        (day, partition, queueWaitTime) = (days[i], partitions[i],
                                           queueWaitTimes[i])
        #Code to make week duration: too slow
        #week = 7
        #weekD = timedelta(7)
        #daysSinceBeg = day - min(days)
        #diff = (daysSinceBeg / week).days
        #day = min(days) + weekD * diff #adjusted date

        if (timeDict.get(day) == None):
            timeDict[day] = {partition : queueWaitTime}
            nJobsDict[day] = {partition : 1}
        elif (timeDict[day].get(partition) == None):
            timeDict[day][partition] = queueWaitTime
            nJobsDict[day][partition] = 1
        else:
            cumWaitTime = timeDict[day][partition]
            nJobs = nJobsDict[day][partition]
            timeDict[day][partition] = cumWaitTime + queueWaitTime
            nJobsDict[day][partition] = nJobs + 1
    return (timeDict, nJobsDict)

def writeHeader(outFile, sep):
    with open(outFile, "w") as f:
        col1 = "Time (day)"
        col2 = "Partition"
        col3 = "Cumulative queue wait time (days)\n"
        f.write("{}{}{}{}{}".format(col1, sep, col2, sep, col3))

def outputData(timeDict, outFile):
    sep = "|"
    writeHeader(outFile, sep)
    with open(outFile, "a") as f:
        for day, partition in timeDict.iteritems():
            for part, td in timeDict[day].iteritems():
               f.write("{}{}{}{}{}\n".format(day, sep, part, sep, td))

def plotTime(timeDict, nJobsDict):
    partitionNames = ["RM", "RM-shared", "GPU", "GPU-shared", "LM"]
    dates = sorted(timeDict.keys())
    fig, (ax, ax2, ax3) = plt.subplots(3)
    for partition in partitionNames:
        times = np.array([timeDict[day].get(partition, 0) for day in dates])
        ax.plot(dates, times, label=partition)

        jobs = np.array([nJobsDict[day].get(partition, 1) for day in dates])
        ax2.plot(dates, jobs, label=partition)

        avgTimes = times/jobs
        ax3.plot(dates, avgTimes, label=partition)

    ax.set(ylabel="Cum Queue Wait Time (days)")
    ax2.set(ylabel="Number of Jobs")
    ax3.set(ylabel="Avg Queue Wait Time (days)")
    fig.autofmt_xdate()
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.75),
               ncol=len(partitionNames))
    plt.show()

#########
#Tests
#########

def testAll():
    return

if __name__ == "__main__":
    start = time.time()
    logsFile = sys.argv[1]
    #outFile = sys.argv[2]
    (days, partitions, queueWaitTimes) = parseLogs(logsFile)
    (tdDict, nJobsDict) = splitDayPartition(days, partitions, queueWaitTimes)
    timeDict= mapTDToTime(tdDict)
    #outputData(timeDict, outFile)
    end = time.time()
    print(end - start)
    plotTime(timeDict, nJobsDict)
    testAll()
