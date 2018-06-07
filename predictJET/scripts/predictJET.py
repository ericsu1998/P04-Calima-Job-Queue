import sys

def processData(logs):
    with open(logs, "r") as f:
        #Remove headers
        lines = f.readlines()[2:]
        #Remove leading and trailing whitespace
        lines = list(map(lambda x: x.strip().split(" "), lines))
        #Remove in-between spaces
        lines = list(map(lambda x: filter(None, x), lines))
    return lines

def extractJET(jobsData):
    #Input: List of data of jobs
    #Output: List of job execution times, in hours

    jobExecTimesStr = [jobData[-1] for jobData in jobsData]

    jobExecTimes = []
    for i in range(len(jobExecTimesStr)):
        jobExecTimeStr = jobExecTimesStr[i]
        daysHMS = jobExecTimeStr.split("-")
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
        jobExecHrs = days*hrsPerDay + hours + minutes/minsPerHr + seconds/secsPerHr
        jobExecTimes.append(jobExecHrs)
    return jobExecTimes

def extractFeatures(jobsData):
    #Input: List of data of jobs
    #Output: List of feature vectors
    return

if __name__ == "__main__":
    logs = sys.argv[1]

    jobsData = processData(logs)
    jobExecTimes = extractJET(jobsData)
    #features = extractFeatures(jobsData)
