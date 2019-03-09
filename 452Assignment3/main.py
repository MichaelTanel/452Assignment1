import csv
import pandas as pd
import numpy as np
from random import uniform
from random import randint
import math
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

numInputNodes = 3
numOutputNodes = 2  # 2 types of clusters

successCount = 0
totalCount = 0
learningRate = 1

# Retrieves data from row
def parseRow(row):
    values = []
    values.append(float(row[1][1]))
    values.append(float(row[1][2]))
    values.append(float(row[1][3]))

    return values

# Object to store the max values from each column
class MaxValues(object):
    xValue = 0
    yValue = 0
    zValue = 0
    
# Normalizes the data by dividing each data point in each column by the columns max value
def normalizeData(maxVals, df):
    df['x_value'] = df['x_value'] / maxVals.xValue
    df['y_value'] = df['y_value'] / maxVals.yValue
    df['z_value'] = df['z_value'] / maxVals.zValue
    
    return df

# Used column headers to easily import the data in columns for more efficient normalizing
def importCSV(filename):
    df = pd.read_csv(filename)

    maxVals = MaxValues()
    maxVals.xValue = max(df['x_value'])
    maxVals.yValue = max(df['y_value'])
    maxVals.zValue = max(df['z_value'])

    return normalizeData(maxVals, df)

# Calculating the new weights using the error correction learning technique
def calculateNewWeights(output, weights, values, expectedOutput):
    learningRate = 0.1
    outputDifference = int(expectedOutput) - int(output)

    # Calculates the new bias weight, with a bias value of 1 
    weights[0] = weights[0] + outputDifference * learningRate * 1

    # Caclulate the new weights for the other criteria
    for i in range(len(values)):
        weights[i + 1] = weights[i + 1] + outputDifference * learningRate * values[i]

    return weights

def calcOutput(weights, values):
    # Initialize an empty list, the same size as the number of hidden nodes
    output = [0 for x in range(len(weights))]

    for i in range(len(weights)):
        sum = 0
        for j in range(len(weights[i])):
            sum += weights[i][j] * values[j]
        # Sigma function
        output[i] = 1 / (1 + math.exp(-sum))

    return output

def calcOutputWeights(hiddenValues, hiddenOutputWeights, deltaJOutput):
    global learningRate    
    newWeights = hiddenOutputWeights.copy()

    # hiddenOutputWeights and deltaJOutput are the same length lists always
    for i in range(len(hiddenOutputWeights)):
        for j in range(len(hiddenOutputWeights[i])):
            newWeights[i][j] = hiddenOutputWeights[i][j] + learningRate * hiddenValues[i] * deltaJOutput[i]

    return newWeights

def train(dfTrain, dfValidate):

    global numInputNodes
    global numOutputNodes
    global numHiddenLayers
    global numHiddenNodes

    # Creates two 2D arrays with floating point numbers between -1 and 1.
    # The first is for the weights between the input nodes and the hidden nodes
    # The second is for the weights between the hidden nodes and the output nodes
    numHiddenNodes = 7
    inputHiddenWeights = [[uniform(-1, 1) for x in range(numInputNodes)] for y in range(numHiddenNodes)]
    hiddenOutputWeights = [[uniform(-1, 1) for x in range(numHiddenNodes)] for y in range(numOutputNodes)]

    weights1 = [uniform(-1, 1) for _ in range(8)]
    weights2 = [uniform(-1, 1) for _ in range(8)]

    # Stores each row of the csv
    values = []
    
    # Wipes the existing file first
    open('output.txt', 'w')
    # Appends to new file
    with open('output.txt', 'a') as outputFile:
        outputFile.write("Initial weights input -> hidden: ")
        outputFile.write(str(inputHiddenWeights))
        outputFile.write("\nInitial weights hidden -> output: ")
        outputFile.write(str(hiddenOutputWeights))
        
    iterations = 200
    trainThreshold = 25

    with open('output.txt', 'a') as outputFile:
        outputFile.write("\nIterations: ")
        outputFile.write(str(iterations))
        termCriteria = "\nTermination criteria: The termination criteria used is the program will run for the number of iterations specified"
        termCriteria += ", or until the success rate is above 90%% for %d runthroughs of the data." % trainThreshold
        outputFile.write(termCriteria)
        layersUsed = "\nNumber of layers used: 3, because there is an input and output layer, as well as 1 hidden layer since the data was not linearly seperable. The reason for only 1 hidden layer is because there is NOT a significant increase in performance if more than one hidden layer is added."
        outputFile.write(layersUsed)
        inputNodes = "\nNumber of input nodes: 9, because there are 9 input features used to classify the glass."
        outputFile.write(inputNodes)
        outputNodes = "\nNumber of output nodes: 6, because there are 6 different types of glass. Since a sigmoid function is being used, there needed to be 1 node per type of output."
        outputFile.write(outputNodes)
        hiddenNodes = "\nNumber of hidden nodes: 7, because there are supposed to be enough nodes to be between the number of input nodes and output nodes."
        outputFile.write(hiddenNodes)

    successRate = 0
    trained = 0

    for i in range(0, iterations):

        # If the successRate is greater than 90% 15 times, the NN is considered to be trained, and
        # the training will stop to prevent overtraining.
        if successRate > 0.9:
            trained += 1
            if (trained == trainThreshold):
                print("Trained:", trained)
                break

        totalCount = 0
        successCount = 0
        successRate = 0
        
        d = {'1': [1,0,0,0,0,0]}
        d['2'] = [0,1,0,0,0,0]
        d['3'] = [0,0,1,0,0,0]
        d['5'] = [0,0,0,1,0,0]
        d['6'] = [0,0,0,0,1,0]
        d['7'] = [0,0,0,0,0,1]

        # Iterate over dataframe row
        for row in dfTrain.iterrows():
            inputValues = parseRow(row)

            # Calculate the outputs at both the hidden layer, and the output layer
            hiddenValues = calcOutput(inputHiddenWeights, inputValues)
            outputValues = calcOutput(hiddenOutputWeights, hiddenValues)

            # The first parameter passed in is the index of the output nodes array with the max value
            # Represented as an array of zeros with one 1. If the 1 is in the third element, the output is 3
            actualGlassType = evaluateGlassType(outputValues.index(max(outputValues)), d)
            expectedGlassType = d[str(inputValues[-1])]


            deltaJOutput = calcDeltaJ(expectedGlassType, outputValues)
            hiddenOutputWeights = calcOutputWeights(hiddenValues, hiddenOutputWeights, deltaJOutput)
            inputHiddenWeights = calcHiddenWeights(inputValues, inputHiddenWeights, hiddenOutputWeights, deltaJOutput, hiddenValues)

            if actualGlassType == expectedGlassType:
                successCount += 1

            totalCount += 1

        successRate = float(successCount) / float(totalCount)

        # Validation
        mse = 0
        mseThreshold = 1
        
        
        # p: number of data points. 6 because there's 6 output nodes
        p = 6 * len(dfValidate.index)

        for row in dfValidate.iterrows():
            inputValues = parseRow(row)

            # Calculate the outputs at both the hidden layer, and the output layer
            hiddenValues = calcOutput(inputHiddenWeights, inputValues)
            outputValues = calcOutput(hiddenOutputWeights, hiddenValues)

            # The first parameter passed in is the index of the output nodes array with the max value
            # Represented as an array of zeros with one 1. If the 1 is in the third element, the output is 3
            actualGlassType = evaluateGlassType(outputValues.index(max(outputValues)), d)
            expectedGlassType = d[str(inputValues[-1])]

            val = np.sum(np.square(np.subtract(expectedGlassType, actualGlassType)))
            mse += float(1 / float(p)) * val
            
            if mse >= mseThreshold:
                return (inputHiddenWeights, hiddenOutputWeights)
            
    return (inputHiddenWeights, hiddenOutputWeights)

def test(inputHiddenWeights, hiddenOutputWeights, dfTest):
    glassTypesDict = {'1': "building_windows_float_processed"}
    glassTypesDict['2'] = "building_windows_non_float_processed"
    glassTypesDict['3'] = "vehicle_windows_float_processed"
    glassTypesDict['5'] = "containers"
    glassTypesDict['6'] = "tableware"
    glassTypesDict['7'] = "headlamps"

    with open('output.txt', 'a') as outputFile:
        outputFile.write("\n\nOriginal\t\t\t\t\t\t\t\tPredicted\n")

    # Lists to hold the expected and actual output.
    # Used when calculating percision and recall.
    expectedOutputList = []
    actualOutputList = []

    totalCount = 0
    successCount = 0

    for row in dfTest.iterrows():
        inputValues = parseRow(row)

        # Calculate the outputs at both the hidden layer, and the output layer
        hiddenValues = calcOutput(inputHiddenWeights, inputValues)
        outputValues = calcOutput(hiddenOutputWeights, hiddenValues)

        actualGlassType = str(outputValues.index(max(outputValues)) + 1)
        if actualGlassType == '4':
            actualGlassType = '5'
        elif actualGlassType == '5':
            actualGlassType = '6'
        elif actualGlassType == '6':
            actualGlassType = '7'

        expectedGlassType = str(inputValues[-1])

        actualOutputList.append(int(actualGlassType))
        expectedOutputList.append(int(expectedGlassType))

        with open('output.txt', 'a') as outputFile:
            outputFile.write("%s" % glassTypesDict[actualGlassType])
            outputFile.write("\t\t\t%s\n" % glassTypesDict[expectedGlassType])

        totalCount += 1

        # If the expectedOutput first bit is equal to the output of the first node
        if actualGlassType == expectedGlassType:
            successCount += 1

        successRate = float(successCount) / float(totalCount)
        # print("Success rate: ", successRate)

    
    print("============================")
    print("Testing")
    print("============================")

    print("Success Count: ", successCount)
    print("Total Count: ", totalCount)
    successRate = float(successCount) / float(totalCount)
    print("Success Rate: %.2f" % successRate)

    return (expectedOutputList, actualOutputList)

def main():
    df = importCSV('dataset_noclass.csv')
    
    inputHiddenWeights, hiddenOutputWeights = train(dfTrain, dfValidate)
        
    percision = precision_score(expectedOutputList, actualOutputList, average='weighted')
    recall = recall_score(expectedOutputList, actualOutputList, average='weighted')
    
    with open('output.txt', 'a') as outputFile:
        outputFile.write("\nFinal weights input -> hidden: ")
        outputFile.write(str(inputHiddenWeights))
        outputFile.write("\nFinal weights hidden -> output: ")
        outputFile.write(str(hiddenOutputWeights))
        outputFile.write("\nPercision score: %.2f\n" % percision)
        outputFile.write("\nRecall score: %.2f\n" % recall)
        outputFile.write("\nConfusion matrix: \n%s" % confusion_matrix(expectedOutputList, actualOutputList))

main()