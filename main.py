import csv
from pprint import pprint as pprint
import pandas as pd
from random import uniform

class Neuron(object):
    def __init__(self):
        self.bias = 1
        self.weightBias = uniform(-1, 1)
        self.area = 0
        self.weightArea = uniform(-1, 1)
        self.perimeter = 0
        self.weightPerimeter = uniform(-1, 1)
        self.compactness = 0
        self.weightCompactness = uniform(-1, 1)
        self.length = 0
        self.weightLength = uniform(-1, 1)
        self.width = 0
        self.weightWidth = uniform(-1, 1)
        self.asymmetryCoefficient = 0
        self.weightAsymmetryCoefficient = uniform(-1, 1)
        self.lengthGroove = 0
        self.weightLengthGroove = uniform(-1, 1)
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
    df['Area'] = df['Area'] / maxVals.area
    df['Perimeter'] = df['Perimeter'] / maxVals.perimeter
    df['Compactness'] = df['Compactness'] / maxVals.compactness
    df['Length'] = df['Length'] / maxVals.length
    df['Width'] = df['Width'] / maxVals.width
    df['AsymmetryCoefficient'] = df['AsymmetryCoefficient'] / maxVals.asymmetryCoefficient
    df['LengthGroove'] = df['LengthGroove'] / maxVals.lengthGroove

    return df

def importCSV(filename):
    df = pd.read_csv(filename)
    maxVals = MaxValues()
    maxVals.area = max(df['Area'])
    maxVals.perimeter = max(df['Perimeter'])
    maxVals.compactness = max(df['Compactness'])
    maxVals.length = max(df['Length'])
    maxVals.width = max(df['Width'])
    maxVals.asymmetryCoefficient = max(df['AsymmetryCoefficient'])
    maxVals.lengthGroove = max(df['LengthGroove'])

    return normalizeData(maxVals, df)

def calculateOutput(neuron):
    output = neuron.bias * neuron.weightBias
    output += neuron.area * neuron.weightArea
    output += neuron.perimeter * neuron.weightPerimeter
    output += neuron.compactness * neuron.weightCompactness
    output += neuron.length * neuron.weightLength
    output += neuron.width * neuron.weightWidth
    output += neuron.asymmetryCoefficient * neuron.weightAsymmetryCoefficient
    output += neuron.lengthGroove * neuron.weightLengthGroove
    return output

errorCount = 0
successCount = 0

def calculateNewWeights(combinedOutput, neuron, expectedOutput):
    learningRate = 0.1

    outputDifference = expectedOutput - int(combinedOutput)

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


def main():
    df = importCSV('trainSeeds.csv')

    neuron1 = Neuron()
    neuron2 = Neuron()

    for i in range(0, 1000):
        print(i)

        # Iterate over dataframe rows
        for row in df.iterrows():
            # In order to get the data point, must access at [1][n], 0 < n < 8
            # This is because row[0] gives the row number, row[1] gives the
            # column name and the data point
            neuron1.area = row[1][0]
            neuron1.perimeter = row[1][1]
            neuron1.compactness = row[1][2]
            neuron1.length = row[1][3]
            neuron1.width = row[1][4]
            neuron1.asymmetryCoefficient = row[1][5]
            neuron1.lengthGroove = row[1][6]
            expectedOutput = row[1][7]

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

            combinedOutput = neuron1.output + neuron2.output    

            # if combinedOutput == "01":
            #     combinedOutput = 1
            # elif combinedOutput == "10":
            #     combinedOutput = 2
            # elif combinedOutput == "11":
            #     combinedOutput = 3

            expectedOutputBinary = format(int(expectedOutput), '02b')
            if expectedOutput[0] != combinedOutput[0]:
                neuron1 = calculateNewWeights(combinedOutput, neuron1)
            elif expectedOutputBinary[1] = neuron1.expectedOutput

            # print(neuron1.weightArea)
            # print(neuron1.weightArea)
            neuron2 = calculateNewWeights(combinedOutput, neuron2)
            # print(neuron2.weightArea)
            # print("---------------------")

            # print(neuron1.weightArea)
        global errorCount
        global successCount
        print(errorCount)
        print(successCount)
        print("Success Rate: ", float(successCount) / float(successCount + errorCount))
        print("-----------------------------------------------")
        errorCount = 0
        successCount = 0

    # print(neuron1.weightArea)
    # successRate = (errorCount * -1) + 16600
    # successRate = float(successRate) / float(16600)
    # print("Success Rate: ", successRate)
main()