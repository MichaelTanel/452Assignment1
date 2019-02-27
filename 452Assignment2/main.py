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

numInputNodes = 9
numOutputNodes = 6  # 6 types of glass
numHiddenLayers = 1
numHiddenNodes = 0

successCount = 0
totalCount = 0
learningRate = 1

# Retrieves data from row
def parseRow(row):
    values = []
    values.append(float(row[1][1]))
    values.append(float(row[1][2]))
    values.append(float(row[1][3]))
    values.append(float(row[1][4]))
    values.append(float(row[1][5]))
    values.append(float(row[1][6]))
    values.append(float(row[1][7]))
    values.append(float(row[1][8]))
    values.append(int(row[1][9]))

    return values

# Object to store the max values from each column
class MaxValues(object):
    refractiveIndex = 0
    sodium = 0
    magnesium = 0
    aluminium = 0
    silicon = 0
    potassium = 0
    calcium = 0
    barium = 0
    iron = 0
    
# Normalizes the data by dividing each data point in each column by the columns max value
def normalizeData(maxVals, df):
    df['Refractive_Index']  = df['Refractive_Index'] / maxVals.refractiveIndex
    df['Sodium']            = df['Sodium'] / maxVals.sodium
    df['Magnesium']         = df['Magnesium'] / maxVals.magnesium
    df['Aluminium']         = df['Aluminium'] / maxVals.aluminium
    df['Silicon']           = df['Silicon'] / maxVals.silicon
    df['Potassium']         = df['Potassium'] / maxVals.potassium
    df['Calcium']           = df['Calcium'] / maxVals.calcium
    df['Barium']            = df['Barium'] / maxVals.barium
    df['Iron']              = df['Iron'] / maxVals.iron

    return df

# Used column headers to easily import the data in columns for more efficient normalizing
def importCSV(filename):
    df = pd.read_csv(filename, usecols=["Refractive_Index", "Sodium", "Magnesium", "Aluminium", "Silicon", "Potassium", "Calcium", "Barium", "Iron", "Glass_Type"])

    maxVals = MaxValues()
    maxVals.refractiveIndex = max(df['Refractive_Index'])
    maxVals.sodium          = max(df['Sodium'])
    maxVals.magnesium       = max(df['Magnesium'])
    maxVals.aluminium       = max(df['Aluminium'])
    maxVals.silicon         = max(df['Silicon'])
    maxVals.potassium       = max(df['Potassium'])
    maxVals.calcium         = max(df['Calcium'])
    maxVals.barium          = max(df['Barium'])
    maxVals.iron            = max(df['Iron'])

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

def evaluateGlassType(value, d):

    if value == 0:
        return d['1']
    elif value == 1:
        return d['2']
    elif value == 2:
        return d['3']
    elif value == 3:
        return d['5']
    elif value == 4:
        return d['6']
    else:
        return d['7']

# Caclulation for deltaJ
def calcDeltaJ(d, y):
    output = [0 for x in range(len(y))]
    for i in range(len(y)):
        output[i] = (d[i] - y[i]) * y[i] * (1 - y[i])

    return output

def calcOutputWeights(hiddenValues, hiddenOutputWeights, deltaJOutput):
    global learningRate    
    newWeights = hiddenOutputWeights.copy()

    # hiddenOutputWeights and deltaJOutput are the same length lists always
    for i in range(len(hiddenOutputWeights)):
        for j in range(len(hiddenOutputWeights[i])):
            newWeights[i][j] = hiddenOutputWeights[i][j] + learningRate * hiddenValues[i] * deltaJOutput[i]

    return newWeights

def calcHiddenWeights(inputValues, inputHiddenWeights, hiddenOutputWeights, deltaJOutput, y):
    global learningRate
    newWeights = inputHiddenWeights.copy()
    output = [0 for x in range(len(hiddenOutputWeights[0]))]

    # Calculate the sum of deltaJ * weights between hidden and output nodes
    for i in range(len(hiddenOutputWeights[0])):
        sum = 0
        for j in range(len(hiddenOutputWeights)):
            sum += deltaJOutput[j] * newWeights[j][i]
        output[i] = sum
    
    # Calculate new weights
    for i in range(len(inputHiddenWeights)):
        tempVal = learningRate * inputValues[i] * output[i] * y[i] * (1 - y[i])
        for j in range(len(inputHiddenWeights[0])):
            newWeights[i][j] = inputHiddenWeights[i][j] + tempVal

    return newWeights

# Split data 70%, 15%, 15%
def splitData(x, y):

    with open('output.txt', 'a') as outputFile:
        str = "\nPercentage of data for training, validation, testing: 70%%, 15%%, 15%%. These numbers were gathered from the notes."
        str += "\nUsing StratifiedShuffleSplit from SciKitLearn, it splits the data and keeps the percentages of the output results the same over the training, validation and testing data."
        str += "\nSo, if 20%% of the output results are a 6, in each of the training, validation, and testing data after the split, 20%% of the output results will be a 6"
        outputFile.write(str)
        
    # Using test_size 0.3 to split the 70% for training and 30% for both validation and test data    
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    xTrainData = []
    yTrainData = []
    xTempData = []
    yTempData = []
    xValidationData = []
    yValidationData = []
    xTestData = []
    yTestData = []
    
    # Split into testing and temp data
    for train_index, temp_index in sss.split(x, y):
        xTrainData, xTempData = x[train_index], x[temp_index]
        yTrainData, yTempData = y[train_index], y[temp_index]
        
    # Using test_size 0.5 to split evenly between the 30% for validation and test data to get 15% of the dataset each
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)

    # Split into validation and testing data
    for validation_index, test_index in sss.split(x[temp_index], y[temp_index]):
        xValidationData, xTestData = x[validation_index], x[test_index]
        yValidationData, yTestData = y[validation_index], y[test_index]
    
    return (xTrainData, yTrainData, xValidationData, yValidationData, xTestData, yTestData)

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
    df = importCSV('GlassData.csv')
    outputData = df['Glass_Type']

    # Converts to numpy array
    data = df.values
    xTrainData, yTrainData, xValidationData, yValidationData, xTestData, yTestData = splitData(data, outputData)

    dfTrain = pd.DataFrame(xTrainData, columns=['Refractive_Index', 'Sodium', 'Magnesium', 'Aluminium', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron', 'Glass_Type'])
    dfValidate = pd.DataFrame(xValidationData, columns=['Refractive_Index', 'Sodium', 'Magnesium', 'Aluminium', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron', 'Glass_Type'])
    dfTest = pd.DataFrame(xTestData, columns=['Refractive_Index', 'Sodium', 'Magnesium', 'Aluminium', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron', 'Glass_Type'])
    
    inputHiddenWeights, hiddenOutputWeights = train(dfTrain, dfValidate)
    
    (expectedOutputList, actualOutputList) = test(inputHiddenWeights, hiddenOutputWeights, dfTest)
    
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