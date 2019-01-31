import csv
import pandas as pd
import numpy as np
from random import uniform
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

successCount = 0
totalCount = 0

# Object to store the max values from each column
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

# Used column headers to easily import the data in columns for more efficient normalizing
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

# Calculates the activation value for the neuron
def calculateActivationValue(values, weights):
    # Bias weight * bias value (1)
    activationValue = weights[0]

    for i in range(len(values)):
        activationValue += values[i] * weights[i + 1]

    return activationValue

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

    return values

# Calculates the output value for each neuron
def calculateOutput(activation):
    return 1 if activation > 0 else 0

def train():
    df = importCSV('trainSeeds.csv')

    global totalCount
    global successCount

    weights1 = []
    weights2 = []
    values = []

    weights1 = [uniform(-1, 1) for _ in range(8)]
    weights2 = [uniform(-1, 1) for _ in range(8)]
    
    # Wipes the existing file first
    open('output.txt', 'w')
    # Appends to new file
    with open('output.txt', 'a') as outputFile:
        outputFile.write("Initial weight 1: ")
        outputFile.write(str(weights1))
        outputFile.write("\nInitial weight 2: ")
        outputFile.write(str(weights2))
        
    iterations = 100
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

        errorCount = 0
        totalCount = 0
        successCount = 0
        successRate = 0

        # Lists to store activation value, output, and expected output 
        errorNeuron1 = []
        errorNeuron2 = []
        
        # Iterate over dataframe row
        for row in df.iterrows():
            values = parseRow(row)
            activation1 = calculateActivationValue(values, weights1)
            activation2 = calculateActivationValue(values, weights2)

            output1 = calculateOutput(activation1)
            output2 = calculateOutput(activation2)

            # values[-1] contains the expected result
            expectedOutputBinary = format(int(values[-1]), '02b')
 
            # If the expectedOutput first bit is equal to the output of the first node
            if int(expectedOutputBinary[0]) != output1:
                weights1copy = weights1
                valuesCopy = values                
                errorNeuron1.append((activation1, output1, weights1copy, valuesCopy, int(expectedOutputBinary[0])))

            # If the expectedOutput second bit is equal to the output of the second node
            if int(expectedOutputBinary[1]) != output2:
                weights2copy = weights2
                valuesCopy = values
                errorNeuron2.append((activation2, output2, weights2copy, valuesCopy, int(expectedOutputBinary[1])))

            # If either of the outputs match their corresponding bit in the expected output,
            # increase the success count.
            if int(expectedOutputBinary[0]) == output1 and int(expectedOutputBinary[1]) == output2:
                successCount += 1

            totalCount += 1

        # print(successCount)
        # print(totalCount - successCount)
        # print(totalCount)
        successRate = float(successCount) / float(totalCount)
        print("Success rate: ", successRate)

        # List not empty
        if len(errorNeuron1) != 0:
            # Sorts to get the closest possible value to 0 as the first element, then use that to adjust the weights            
            errorNeuron1.sort(key=lambda tup: abs(tup[0]))  # sorts in place
            weights1 = calculateNewWeights(errorNeuron1[0][1], errorNeuron1[0][2], errorNeuron1[0][3], errorNeuron1[0][4])

        # List not empty
        if len(errorNeuron2) != 0:
            # Sorts to get the closest possible value to 0 as the first element, then use that to adjust the weights
            errorNeuron2.sort(key=lambda tup: abs(tup[0]))  # sorts in place
            weights2 = calculateNewWeights(errorNeuron2[0][1], errorNeuron2[0][2], errorNeuron2[0][3], errorNeuron2[0][4])

        print("--------------------------------------------")

    return (weights1, weights2)

def test(weights1, weights2):
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

def main():
    (weights1, weights2) = train()
    (expectedOutputList, actualOutputList) = test(weights1, weights2)
    with open('output.txt', 'a') as outputFile:
        outputFile.write("\nFinal weight 1: ")
        outputFile.write(str(weights1))
        outputFile.write("\nFinal weight 2: ")
        outputFile.write(str(weights2))

    externalToolTraining(expectedOutputList, actualOutputList)

# Training perceptron using Scikit
def externalToolTraining(expectedOutputList, actualOutputList):
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

    # Calculating percision and recall of my code
    (tn, fp, fn, tp) = confusion_matrix(actualOutputList, expectedOutputList)
    percision = float(tp) / (float(tp) + float(fp))
    recall = float(tp) / (float(tp) + float(fn))

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
        outputFile.write("My code: %.2f" % recall)

main()