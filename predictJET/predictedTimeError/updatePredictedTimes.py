import sys
import os.path
from datetime import datetime, timedelta

sys.path.insert(0, '/pylon5/cc5fp8p/esu1/projects/p04/predictJET/torchNN')
from misc import save, load

def extractData(logFile, outFile):
    #Extracts JobID and predicted times
    #Input:
    #   logFile: File to extract jobs from
    #            Format should be:
    #            JobID  startTime
    #            <job1> <time1>
    #            <job2> <time2>
    #            ...
    #   outFile: File to store extracted jobs

    estStartTimes = load(outFile) if os.path.isfile(outFile) else dict()
    if (len(estStartTimes) != 0):
        print("N (previous):", len(estStartTimes))

    with open(logFile, "r") as f:
        for line in f:
            parsedLine = line.strip().split(" ")
            if (len(parsedLine) == 2):
                (jobID, estStartTime) = parsedLine
                timeFormat = "%Y-%m-%dT%H:%M:%S"
                estStartTime = datetime.strptime(estStartTime,
                                                 timeFormat)
                if (jobID not in estStartTimes):
                    estStartTimes[jobID] = [estStartTime, None, None]

    print("N:", len(estStartTimes))
    save(estStartTimes, outFile)

if __name__ == "__main__":
    logFile = sys.argv[1]
    outFile = sys.argv[2]
    extractData(logFile, outFile)
