import time
import pickle

class TimeTracker:
    def __init__(self):
        self.time = time.time()
        self.descriptor = None

    def printTimeElapsed(self):
        newTime = time.time()
        timeElapsed = newTime - self.time
        if (self.descriptor != None):
            print("{0} took {1:.2f} seconds".format(self.descriptor, timeElapsed))
        else:
            print("{0:.2f} seconds have elapsed since last call".format(timeElapsed))

    def startBlock(self, descriptor):
        self.descriptor = descriptor
        print("{}...".format(self.descriptor))
        self.time = time.time()

    def endBlock(self):
        self.printTimeElapsed()
        if (self.descriptor != None):
            print("{} finshed!\n".format(self.descriptor))
        self.descriptor = None

    def changeBlock(self, descriptor):
        self.endBlock()
        self.startBlock(descriptor)

def save(L, outFile):
    #Saving lists
    pickleOut = open(outFile, "wb")
    pickle.dump(L, pickleOut)
    pickleOut.close()

def load(inFile):
    #Loading lists
    pickleIn = open(inFile, "rb")
    return pickle.load(pickleIn)


