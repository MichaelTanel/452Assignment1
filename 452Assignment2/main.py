import csv
import pandas as pd
import numpy as np
from random import uniform
from random import randint
import math

numInputNodes = 9
numOutputNodes = 6  # 6 types of glass
numHiddenLayers = 1
numHiddenNodes = 0

successCount = 0
totalCount = 0

# Retrieves data from row
def parseRow(row):
    values = []
    # values.append(float(row[1][1]))
    # Skip row[1][1] since it's the ID
    values.append(float(row[1][2]))
    values.append(float(row[1][3]))
    values.append(float(row[1][4]))
    values.append(float(row[1][5]))
    values.append(float(row[1][6]))
    values.append(float(row[1][7]))
    values.append(float(row[1][8]))
    values.append(float(row[1][9]))
    values.append(int(row[1][10]))
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
    df = pd.read_csv(filename)

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
        output[i] = (int(d[i]) - int(y[i])) * int(y[i]) * (1 - int(y[i]))

    return output

# y and d are actual and expected values arrays respectively
def calcOutputWeights(hiddenValues, hiddenOutputWeights, d, y, deltaJOutput):
    
    learningRate = 0.05
    newWeights = hiddenOutputWeights.copy()

    # hiddenOutputWeights and deltaJOutput are the same length lists always
    for i in range(len(hiddenOutputWeights)):
        for j in range(len(hiddenOutputWeights[i])):
            newWeights[i][j] = learningRate * hiddenValues[i] * deltaJOutput[i]

    return newWeights

def calcHiddenWeights():
    num = 1

# Split data 70%, 15%, 15%
def train():
    df = importCSV('GlassData.csv')

    global totalCount
    global successCount
    global numInputNodes
    global numOutputNodes
    global numHiddenLayers
    global numHiddenNodes

    # numHiddenNodes = randint(numOutputNodes, numInputNodes)

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
        
    iterations = 2000
    trainThreshold = 25

    with open('output.txt', 'a') as outputFile:
        outputFile.write("\nIterations: ")
        outputFile.write(str(iterations))
        termCriteria = "\nTermination criteria: The termination criteria used is the program will run for the number of iterations specified"
        termCriteria += ", or until the success rate is above 90%% for %d runthroughs of the data." % trainThreshold
        outputFile.write(termCriteria)

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
        
        # Iterate over dataframe row
        for row in df.iterrows():
            inputValues = parseRow(row)

            # Calculate the outputs at both the hidden layer, and the output layer
            hiddenValues = calcOutput(inputHiddenWeights, inputValues)
            outputValues = calcOutput(hiddenOutputWeights, hiddenValues)

            d = {'1': "100000"}
            d['2'] = "010000"
            d['3'] = "001000"
            d['5'] = "000100"
            d['6'] = "000010"
            d['7'] = "000001"

            # The first parameter passed in is the index of the output nodes array with the max value
            actualGlassType = evaluateGlassType(outputValues.index(max(outputValues)), d)
            expectedGlassType = d[str(inputValues[-1])]

            # If the expectedOutput first bit is equal to the output of the first node
            if actualGlassType  == expectedGlassType:
                successCount += 1

            totalCount += 1

            deltaJOutput = calcDeltaJ(expectedGlassType, actualGlassType)
            hiddenOutputWeights = calcOutputWeights(hiddenValues, hiddenOutputWeights, expectedGlassType, actualGlassType, deltaJOutput)
            calcHiddenWeights()

        successRate = float(successCount) / float(totalCount)
        print("Success rate: ", successRate)

    return (inputHiddenWeights, hiddenOutputWeights)

def test(weights1, weights2):

    # TODO: output initial weights, node output function used
    # learning rate, termination criteria & proper explanations for the choice

    df = importCSV('testSeeds.csv')

    successTestCount = 0
    totalCount = 0

    with open('output.txt', 'a') as outputFile:
        outputFile.write("\n\nOriginal\tPredicted\n")

    # Lists to hold the expected and actual output.
    # Used when calculating percision and recall.
    expectedOutputList = []
    actualOutputList = []

    # Iterate over dataframe row
    for row in df.iterrows():

        values = parseRow(row)
        activation1 = calculateActivationValue(values, weights1)
        activation2 = calculateActivationValue(values, weights2)

        output1 = calculateOutput(activation1)
        output2 = calculateOutput(activation2)

        # values[-1] contains the expected result
        expectedOutputBinary = format(int(values[-1]), '02b')

        # If either of the outputs did not match their corresponding bit in the expected output,
        # increase the error.
        if int(expectedOutputBinary[0]) == output1 and int(expectedOutputBinary[1]) == output2:
            successTestCount += 1

        # Convert 2 outputs to decimal value
        if output1 == 0 and output2 == 1:
            expectedOutputList.append(1)
        elif output1 == 1 and output2 == 0:
            expectedOutputList.append(2)
        elif output1 == 1 and output2 == 1:
            expectedOutputList.append(3)
        else:
            expectedOutputList.append(0)
        
        actualOutputList.append(int(values[-1]))

        with open('output.txt', 'a') as outputFile:
            output = output1 + output2
            outputFile.write("%d" % int(values[-1]))
            outputFile.write("\t\t\t%s\n" % output)

        totalCount += 1
    
    print("============================")
    print("Testing")
    print("============================")

    print("Success Count: ", successTestCount)
    print("Total Count: ", totalCount)
    successRate = float(successTestCount) / float(totalCount)
    print("Success Rate: %.2f", successRate)

    return (expectedOutputList, actualOutputList)

# Training perceptron using Scikit
def externalToolTraining(percision, recall):
    # Added skip rows due to the addition of headers in the csvs.
    trainingData = np.loadtxt('trainSeeds.csv', delimiter=',', skiprows=1)
    testData = np.loadtxt('testSeeds.csv', delimiter=',', skiprows=1)

    # Removing last column
    trainingInputData = trainingData[:, :-1]
    # Removing all columns except last column
    trainingDesiredOutput = trainingData[:, -1]

    # Removing last column
    testInputData = testData[:, :-1]
    # Removing all columns except last column
    testDesiredOutput = testData[:, -1]

    # Using scikit learn
    ss = StandardScaler()
    ss.fit(trainingInputData)
    train = ss.transform(trainingInputData)
    test = ss.transform(testInputData)
    perceptron = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    perceptron.fit(train, trainingDesiredOutput)

    prediction = perceptron.predict(test)

    open('toolBasedOutput.txt', 'w')
    with open('toolBasedOutput.txt', 'a') as outputFile:
        outputFile.write("Percision\n")
        outputFile.write("--------------------------\n")
        outputFile.write("Scikit Learn: %.2f\n" % precision_score(testDesiredOutput, prediction, average='weighted'))
        outputFile.write("My code: %.2f\n" % percision)
        outputFile.write("--------------------------\n")
        outputFile.write("Recall\n")
        outputFile.write("--------------------------\n")
        outputFile.write("Scikit Learn: %.2f\n" % recall_score(testDesiredOutput, prediction, average='weighted'))
        outputFile.write("My code: %.2f\n" % recall)

def main():
    inputHiddenWeights, hiddenOutputWeights = train()
    # (expectedOutputList, actualOutputList) = test(weights1, weights2)
    
    # percision = precision_score(expectedOutputList, actualOutputList, average='weighted')
    # recall = recall_score(expectedOutputList, actualOutputList, average='weighted')
    # 
    # with open('output.txt', 'a') as outputFile:
        # outputFile.write("\nFinal weight 1: %s" % str(weights1))
        # outputFile.write("\nFinal weight 2: %s\n" % str(weights2))
        # outputFile.write("Percision score: %.2f\n" % percision)
        # outputFile.write("Recall score: %.2f\n" % recall)
        # outputFile.write("\nConfusion matrix: \n%s" % confusion_matrix(expectedOutputList, actualOutputList))
# 
    # externalToolTraining(percision, recall)

main()