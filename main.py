import csv
from pprint import pprint as pprint
import pandas as pd
from random import uniform

class Neuron(object):
    def __init__(self):
        self.bias = 1.0
        self.weightBias = uniform(-1, 1)
        # self.weightBias = 1000
        self.area = 0.0
        self.weightArea = uniform(-1, 1)
        # self.weightArea = 1000
        self.perimeter = 0.0
        self.weightPerimeter = uniform(-1, 1)
        # self.weightPerimeter = 1000
        self.compactness = 0.0
        self.weightCompactness = uniform(-1, 1)
        # self.weightCompactness = 1000
        self.length = 0.0
        self.weightLength = uniform(-1, 1)
        # self.weightLength = 1000
        self.width = 0.0
        self.weightWidth = uniform(-1, 1)
        # self.weightWidth = 1000
        self.asymmetryCoefficient = 0.0
        self.weightAsymmetryCoefficient = uniform(-1, 1)
        # self.weightAsymmetryCoefficient = 1000
        self.lengthGroove = 0.0
        self.weightLengthGroove = uniform(-1, 1)
        # self.weightLengthGroove = 1000
        self.activationValue = 0.0
        self.output = 0

class MaxValues(object):
    area = 0
    perimeter = 0
    compactness = 0
    length = 0
    width = 0
    asymmetryCoefficient = 0
    lengthGroove = 0

# Normalizes the data by dividing each data point in each column by the columns max value
def normalizeData(maxVals, df):
    df['Area']                  = df['Area'] / maxVals.area
    df['Perimeter']             = df['Perimeter'] / maxVals.perimeter
    df['Compactness']           = df['Compactness'] / maxVals.compactness
    df['Length']                = df['Length'] / maxVals.length
    df['Width']                 = df['Width'] / maxVals.width
    df['AsymmetryCoefficient']  = df['AsymmetryCoefficient'] / maxVals.asymmetryCoefficient
    df['LengthGroove']          = df['LengthGroove'] / maxVals.lengthGroove
    return df

def importCSV(filename):
    df = pd.read_csv(filename)
    maxVals = MaxValues()
    maxVals.area                    = max(df['Area'])
    maxVals.perimeter               = max(df['Perimeter'])
    maxVals.compactness             = max(df['Compactness'])
    maxVals.length                  = max(df['Length'])
    maxVals.width                   = max(df['Width'])
    maxVals.asymmetryCoefficient    = max(df['AsymmetryCoefficient'])
    maxVals.lengthGroove            = max(df['LengthGroove'])

    return normalizeData(maxVals, df)

# Calculates the output, 0 or 1, for the neuron
def calculateOutput(neuron):
    activationValue = neuron.bias * neuron.weightBias
    activationValue += neuron.area * neuron.weightArea
    activationValue += neuron.perimeter * neuron.weightPerimeter
    activationValue += neuron.compactness * neuron.weightCompactness
    activationValue += neuron.length * neuron.weightLength
    activationValue += neuron.width * neuron.weightWidth
    activationValue += neuron.asymmetryCoefficient * neuron.weightAsymmetryCoefficient
    activationValue += neuron.lengthGroove * neuron.weightLengthGroove
    
    neuron.activationValue = activationValue
    neuron.output = 1 if activationValue >= 0 else 0
    return neuron

errorCount = 0
successCount = 0

def calculateNewWeights(actual, neuron, expectedOutput):
    learningRate = 0.5

    outputDifference = int(expectedOutput) - int(actual)

    if outputDifference == 0:
        global successCount
        successCount += 1
    else:
        global errorCount
        errorCount += 1

    neuron.weightBias                   = neuron.weightBias + outputDifference * learningRate * neuron.bias
    neuron.weightArea                   = neuron.weightArea + outputDifference * learningRate * neuron.area
    neuron.weightPerimeter              = neuron.weightPerimeter + outputDifference * learningRate * neuron.perimeter
    neuron.weightCompactness            = neuron.weightCompactness + outputDifference * learningRate * neuron.compactness
    neuron.weightLength                 = neuron.weightLength + outputDifference * learningRate * neuron.length
    neuron.weightWidth                  = neuron.weightWidth + outputDifference * learningRate * neuron.width
    neuron.weightAsymmetryCoefficient   = neuron.weightAsymmetryCoefficient + outputDifference * learningRate * neuron.asymmetryCoefficient
    neuron.weightLengthGroove           = neuron.weightLengthGroove + outputDifference * learningRate * neuron.lengthGroove

    return neuron

# Retrieves data from row
def parseRow(neuron, row):
    neuron.area = float(row[1][0])
    neuron.perimeter = float(row[1][1])
    neuron.compactness = float(row[1][2])
    neuron.length = float(row[1][3])
    neuron.width = float(row[1][4])
    neuron.asymmetryCoefficient = float(row[1][5])
    neuron.lengthGroove = float(row[1][6])

    return neuron

def main():
    df = importCSV('trainSeeds.csv')

    neuron1 = Neuron()
    neuron2 = Neuron()

    errorNeuron1 = []
    errorNeuron2 = []
    for i in range(0, 1):
        # Iterate over dataframe rows
        for row in df.iterrows():
            neuron1 = parseRow(neuron1, row)
            print(neuron1.area)
            neuron1 = calculateOutput(neuron1)

            expectedOutput = row[1][7]

            neuron2 = parseRow(neuron2, row)
            neuron2 = calculateOutput(neuron2)

            expectedOutputBinary = format(int(expectedOutput), '02b')
            
            # If the expectedOutput is equal to the output of the first node
            if int(expectedOutputBinary[0]) != neuron1.output:
                errorNeuron1.append(neuron1)
                # neuron1 = calculateNewWeights(neuron1.output, neuron1, int(expectedOutputBinary[0]))
            elif int(expectedOutputBinary[1]) != neuron2.output:
                errorNeuron2.append(neuron2)
                # neuron2 = calculateNewWeights(neuron1.output, neuron2, int(expectedOutputBinary[1]))
        
        errorNeuron1.sort(key=lambda x: x.activationValue)
        print(len(errorNeuron1))
        print(errorNeuron1[0].activationValue)
        print(errorNeuron1[54].activationValue)

        global errorCount
        global successCount
        # print(errorCount)
        # print(successCount)
        # print("Success Rate: ", float(successCount) / float(successCount + errorCount))
        print("-----------------------------------------------")
        errorCount = 0
        successCount = 0

    # print(neuron1.weightArea)

    df = importCSV('testSeeds.csv')

    successTestCount = 0
    errorTestCount = 0

    countZeroOne = 0
    countOneZero = 0
    countOneOne = 0

    for row in df.iterrows():
        neuron1.area = row[1][0]
        neuron1.perimeter = row[1][1]
        neuron1.compactness = row[1][2]
        neuron1.length = row[1][3]
        neuron1.width = row[1][4]
        neuron1.asymmetryCoefficient = row[1][5]
        neuron1.lengthGroove = row[1][6]
        expectedOutput = int(row[1][7])

        neuron1.output = calculateOutput(neuron1)

        if neuron1.output > 0:
            neuron1.output = "1"
        else:
            neuron1.output = "0"

        neuron2.area = row[1][0]
        neuron2.perimeter = row[1][1]
        neuron2.compactness = row[1][2]
        neuron2.length = row[1][3]
        neuron2.width = row[1][4]
        neuron2.asymmetryCoefficient = row[1][5]
        neuron2.lengthGroove = row[1][6]

        neuron2.output = calculateOutput(neuron2)

        if neuron2.output > 0:
            neuron2.output = "1"
        else:
            neuron2.output = "0"

        combinedOutputBinary = neuron1.output + neuron2.output    
        # print(combinedOutputBinary)
        combinedOutput = 0
        if combinedOutputBinary == "01":
            countZeroOne += 1
            # print("in 01")
            combinedOutput = 1
        elif combinedOutputBinary == "10":
            countOneZero += 1
            # print("in 10")
            combinedOutput = 2
        elif combinedOutputBinary == "11":
            countOneOne += 1
            # print("in 11")
            combinedOutput = 3
        
        if expectedOutput == combinedOutput:
            successTestCount += 1
        else:
            errorTestCount += 1
    print("===================================================")
    print("Success: ", successTestCount)
    print("Error: ", errorTestCount)
    print("01: ", countZeroOne)
    print("10: ", countOneZero)
    print("11: ", countOneOne)
    print("Success rate: ", float(successTestCount) / (successTestCount + errorTestCount))
    # print("Success Rate: ", successRate)
main()