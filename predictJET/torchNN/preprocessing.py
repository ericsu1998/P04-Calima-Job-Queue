import csv
import numpy as np
import torch

def getHeaderNames(csvFile, sep=","):
    reader = csv.reader(open(csvFile, newline=''), delimiter=sep)
    return next(reader)

def preprocess(csvFile, sep=",", header=False,
               featuresMapFns=None, labelMapFn = None):
    #Extracts features and labels from a CSV file
    #Input:
    #   csvFile = CSV file to extract features and labels from
    #   sep = What CSV file is separated by
    #   header = If True, skips the header of the CSV file
    #   featuresF = List of mapping functions for each feature
    #               (None if no mapping desired)
    #   labelsF = mapping function for labels
    #             (None if no mapping desired)
    #Output:
    #   List of features and list of labels

    features = []
    labels = []

    reader = csv.reader(open(csvFile, newline=''), delimiter=sep)
    if header: next(reader)

    for line in reader:
        for i in range(len(line)):
            if (line[i] == ""):
                #Missing value: impute w/ previous value
                line[i] = features[-1][i] if i < len(line) - 1 else labels[-1]
            else:
                if (i < len(line) - 1 and featuresMapFns != None):
                    line[i] = featuresMapFns[i](line[i])
                elif (i == len(line) - 1 and labelMapFn != None):
                    line[i] = labelMapFn(line[i])

        features.append(line[:-1])
        labels.append(line[-1])

    return features, labels

def getMapping(L):
    #Input: L = list
    #Output: Maps each unique element in list to a number
    S = set(L)
    mapping = dict()
    count = 0
    for item in S:
        mapping[item] = count
        count += 1
    return mapping

def getMappings(L, nCat):
    #Inputs:
        #L: List of features
        #nCat: number of categorical features to map
    #Output:
        #List of n mappings

    mappings = []
    L = np.array(L)
    for col in range(nCat):
        catColumn = [L[row][col] for row in range(len(L))]
        mappings.append(getMapping(catColumn))

    return mappings

def getOffsets(n, D=dict()):
    #Input:
        #n = Number of total Features
        #D = List of Dictionaries of categorical mappings
        #Assumptions: len(D) < n, n > 0
    #Output: Offsets for a sparse one-hot-vector representation of a matrix,
    #        with the categorical features first

    offset = 0
    offsets = []
    nCatFeatures = len(D)
    for i in range(n):
        offsets.append(offset)
        offset += len(D[i]) if i < nCatFeatures else 1

    endOffset = offsets[-1] + len(D[-1]) if nCatFeatures == n else offsets[-1] + 1
    return (offsets, endOffset)

def toSparseTensor(L, D):
    #Input:
        #L = list to convert into sparse tensor
        #D = List of Dictionaries of categorical mappings
    #Assumptions: L is nonempty, all categorical features in L are before
    #             any quantitative features
    #Output: Sparse tensor

    rows = []
    cols = []
    values = []

    N = len(L)
    nCatFeatures = len(D)
    (offsets, nSparseFeatures) = getOffsets(len(L[0]), D)

    for row in range(N):
        featureVec = L[row]
        for i in range(len(featureVec)):
            if (i < nCatFeatures):
                catFeature = featureVec[i]
                col = offsets[i] + D[i][catFeature]
                value = 1
            else:
                col = offsets[i]
                value = featureVec[i]

            rows.append(row)
            cols.append(col)
            values.append(value)

    indices = torch.LongTensor([rows, cols])
    values = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(indices, values, torch.Size([N, nSparseFeatures]))

#Checking/Debugging Methods
def hasMissingValues(L):
    for item in L:
        if (item == "" or (isinstance(item, list) and ("" in item))):
            return True
    return False
#End of Checking/Debugging Methods

def testHasMissingValues():
    assert(hasMissingValues([""]))
    assert(hasMissingValues([[""]]))
    assert(hasMissingValues([2,3,"",5,6]))
    assert(not hasMissingValues([2,3,4]))
    assert(not hasMissingValues([[1,2,3], ['a','b','c']]))

def testGetMappings():
    L1 = [['a'], ['a'], ['c'], ['d'], ['c'], ['b']]
    print(getMappings(L1, nCat=1))
    L2 = [['a', 2], ['a', 1], ['c', 3], ['d', 5], ['c', 6], ['b', 7]]
    print(getMappings(L2, nCat=1))
    print(getMappings(L2, nCat=2))

def testGetOffsets():
    n1 = 1
    assert(getOffsets(n1) == ([0], 1))

    n1 = 2
    D1 = [{'a':0, 'b':1, 'c':2}]
    assert(getOffsets(n1, D1) == ([0, 3], 4))

    n2 = 5
    D2 = [{'a':0}, {'a':0, 'b':1}, {'a':0, 'b':1, 'c':2}]
    assert(getOffsets(n2, D2) == ([0, 1, 3, 6, 7], 8))

    n3 = 2
    D3 = [{'a':0}, {'a':0, 'b':1}]
    assert(getOffsets(n3, D3) == ([0,1], 3))

def testToSparseTensor():
    L1 = [['a', 1], ['c', 2], ['b', 3]]
    D1 = [{'a':0, 'b':1, 'c':2}]
    T1 = toSparseTensor(L1, D1)
    assert(T1._indices().tolist() == [[0,0,1,1,2,2],[0,3,2,3,1,3]])
    assert(T1._values().tolist() == [1.0,1.0,1.0,2.0,1.0,3.0])

def testAll():
    testHasMissingValues()
    testGetMappings()
    testGetOffsets()
    testToSparseTensor()

if __name__ == "__main__":
    testAll()
