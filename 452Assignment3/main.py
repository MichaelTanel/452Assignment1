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
from shutil import copyfile

numInputNodes = 3
numOutputNodes = 2  # 2 types of clusters

successCount = 0
totalCount = 0
learningRate = 0.5
iterations = 10

# Retrieves data from row
def parseRow(row):
    values = []
    values.append(float(row[1][0]))
    values.append(float(row[1][1]))
    values.append(float(row[1][2]))

    return values

# Used column headers to easily import the data in columns for more efficient normalizing
def importCSV(filename):
    return pd.read_csv(filename)

def calcNet(inputValues, weights1, weights2):
    net = [0] * 2
    for i in range(len(weights1)):
        net[0] += weights1[i] * inputValues[i]
        net[1] += weights2[i] * inputValues[i]

    return net
    
def findWinner(net):
    maxVal = [0] * 2
    maxVal[0] = max(0, net[0] - (1 / 2) * net[1])
    maxVal[1] = max(0, net[1] - (1 / 2) * net[0])

    if np.abs(maxVal[0]) < 1e-10:
        maxVal[0] = 0
    
    if np.abs(maxVal[1]) < 1e-10:
        maxVal[1] = 0

    # Update values
    while np.count_nonzero(maxVal) > 1:
        maxVal[0] = max(0, maxVal[0] - (1 / 2) * maxVal[1])
        maxVal[1] = max(0, maxVal[1] - (1 / 2) * maxVal[0])
        
        if np.abs(maxVal[0]) < 1e-10:
            maxVal[0] = 0
    
        if np.abs(maxVal[1]) < 1e-10:
            maxVal[1] = 0


    if maxVal[0] == 0:
        return 0
    elif maxVal[1] == 0:
        return 1
    else:
        return 2    # No winner

def updateWeight(inputValues, weights):
    global learningRate
    weights = weights + learningRate * np.subtract(inputValues, weights)

    return weights

def calcError(inputValues, weights1, weights2):
    error = 0
    for i in range(len(inputValues)):
        error += (inputValues[i] - weights1[i]) ** 2
        error += (inputValues[i] - weights2[i]) ** 2

    return error

def kohonen(df):

    global numInputNodes
    global numOutputNodes
    global iterations

    # Creates two 1D arrays with floating point numbers between -1 and 1.
    # weights1 represents the weights for the connections from all input nodes to output node 1.
    # weights2 represents the weights for the connections from all input nodes to output node 2.
    weights1 = [uniform(-1, 1) for x in range(numInputNodes)]
    weights2 = [uniform(-1, 1) for x in range(numInputNodes)]
    
    # Wipes the existing file first
    open('output.txt', 'w')
    # Appends to new file
    with open('output.txt', 'a') as outputFile:
        outputFile.write("Initial weights: ")
        outputFile.write(str(weights1))
        outputFile.write("\n")
        outputFile.write(str(weights2))
        # TODO: fix term criteria
        termCriteria = "\nTermination criteria: The termination criteria used is the program will run for however many rows of data there are."
        outputFile.write(termCriteria)
        inputNodes = "\nNumber of input nodes: 3, because there are 3 inputs."
        outputFile.write(inputNodes)
        outputNodes = "\nNumber of output nodes: 2, because there are 2 different clusters."
        outputFile.write(outputNodes)

    # Stores the results of the clustering function
    results = [0] * len(df.index)
    prevCalcError = 0

    for i in range(iterations):
        j = 0
        # Iterate over dataframe row
        for row in df.iterrows():
            inputValues = parseRow(row)
            
            net = calcNet(inputValues, weights1, weights2)

            winner = findWinner(net)

            if winner == 0:
                weights1 = updateWeight(inputValues, weights1)
                results[j] = winner + 1
            elif winner == 1:
                weights2 = updateWeight(inputValues, weights2)
                results[j] = winner + 1
            else:
                # Randomly choose a winner
                results[j] = random.randint(1, 2)
            
            j += 1

        # If it is the first iteration, there's no previous error calc to compare it to.
        # If it is not the first iteration, calculate the error and compare it to the previous error.
        # If the new error is greater than previous error, stop looping.
        if i == 0:
            prevCalcError = calcError(inputValues, weights1, weights2)
        else:
            newCalcError = calcError(inputValues, weights1, weights2)

            if newCalcError > prevCalcError:
                print(i)
                break
            prevCalcError = newCalcError

    # copyfile("dataset_noclass.csv", "kohonen_results.csv")
    dfNew = pd.read_csv('dataset_noclass.csv')
    # Removes the first line (headers)
    # dfNew = dfNew.iloc[1:]
    dfNew['output'] = results
    dfNew.to_csv('kohonen_results.csv')
        
    return (weights1, weights2, results)

def calcDistance(inputValues, prototype1, prototype2):
    distance1 = math.sqrt((prototype1[0] - inputValues[0]) ** 2 + (prototype1[1] - inputValues[1]) ** 2 + (prototype1[2] - inputValues[2]) ** 2)
    distance2 = math.sqrt((prototype2[0] - inputValues[0]) ** 2 + (prototype2[1] - inputValues[1]) ** 2 + (prototype2[2] - inputValues[2]) ** 2)

    if distance1 < distance2:
        return 1
    elif distance1 > distance2:
        return 2
    else:
        return randint(1, 2)

# Updates the values of the prototypes by summing the list of input vectors, and dividing them by the cluster size.
def updatePrototype(cluster1, cluster2):
    
    size1 = len(cluster1)
    size2 = len(cluster2)

    # Calculate new prototype values for both clusters
    prototype1 = [sum(x) for x in zip(*cluster1)]
    prototype1 = np.true_divide(prototype1, size1)

    prototype2 = [sum(x) for x in zip(*cluster2)]
    prototype2 = np.true_divide(prototype2, size2)

    return prototype1, prototype2

def calcKMeansError(inputValues, prototype1, prototype2):
    error = np.sum(np.square(np.subtract(inputValues, prototype1)))
    error += np.sum(np.square(np.subtract(inputValues, prototype2)))
    return error

# Implementing the k-means algorithm
def kMeans(df):

    global iterations
    # Stores the results of the clustering function
    results = [0] * (len(df.index) - 2)
    prevCalcError = 0

    # Retrive 2 random points as the basiss for a "cluster"
    prototype1Index = randint(0, len(df.index) - 1)
    prototype2Index = randint(0, len(df.index) - 1)

    prototype1 = [0] * 3
    prototype2 = [0] * 3

    # Gets each x, y, z value into array
    rowVal = df.iloc[prototype1Index]
    prototype1[0], prototype1[1], prototype1[2] = rowVal[0], rowVal[1], rowVal[2]

    rowVal = df.iloc[prototype2Index]
    prototype2[0], prototype2[1], prototype2[2] = rowVal[0], rowVal[1], rowVal[2]

    cluster1 = []
    cluster2 = []

    for i in range(iterations):
        j = 0
        # Iterate over dataframe row
        for row in df.iterrows():
            inputValues = parseRow(row)
            
            result = calcDistance(inputValues, prototype1, prototype2)
            
            if result == 1:
                cluster1.append(inputValues)
            # result == 2
            else:
                cluster2.append(inputValues)

            j += 1

        prototype1, prototype2 = updatePrototype(cluster1, cluster2)

        # If it is the first iteration, there's no previous error calc to compare it to.
        # If it is not the first iteration, calculate the error and compare it to the previous error.
        # If the new error is greater than previous error, stop looping.
        if i == 0:
            prevCalcError = calcKMeansError(inputValues, prototype1, prototype2)
        else:
            newCalcError = calcKMeansError(inputValues, prototype1, prototype2)
            # If the 
            if newCalcError > prevCalcError:
                break
            prevCalcError = newCalcError

    return results


def main():
    df = importCSV('dataset_noclass.csv')
    
    weights1, weights2, results = kohonen(df)
    kMeans(df)

    # percision = precision_score(expectedOutputList, actualOutputList, average='weighted')
    # recall = recall_score(expectedOutputList, actualOutputList, average='weighted')
    
    with open('output.txt', 'a') as outputFile:
        outputFile.write("\nFinal weights: ")
        outputFile.write(str(weights1))
        outputFile.write(str(weights2))
        # outputFile.write("\nPercision score: %.2f\n" % percision)
        # outputFile.write("\nRecall score: %.2f\n" % recall)
        # outputFile.write("\nConfusion matrix: \n%s" % confusion_matrix(expectedOutputList, actualOutputList))

main()