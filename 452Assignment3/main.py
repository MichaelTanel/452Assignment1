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
learningRate = 0.5

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

    while (maxVal[0] == 0 and maxVal[1] == 0) or (maxVal[0] != 0 and maxVal[1] != 0):
        maxVal[0] = max(0, net[0] - (1 / 2) * net[1])
        maxVal[1] = max(0, net[1] - (1 / 2) * net[0])

    if maxVal[0] == 0:
        return 0
    else:
        return 1

def updateWeight(inputValues, weights):
    global learningRate
    weights = weights + learningRate * np.subtract(inputValues, weights)

    return weights

def cluster(df):

    global numInputNodes
    global numOutputNodes

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

    results = [0] * len(df.index)
    i = 0
    print("hello")
    # Iterate over dataframe row
    for row in df.iterrows():
        inputValues = parseRow(row)
        print("------------------------")
        print(inputValues)
        print(weights1)
        print(weights2)
        net = calcNet(inputValues, weights1, weights2)

        print("findwinner")
        winner = findWinner(net)
        print("foundwinner")
        if winner == 0:
            weights1 = updateWeight(inputValues, weights1)
        else:
            weights2 = updateWeight(inputValues, weights2)

        # Store the result in an array. The +1 is because winner can be 1 or 0
        # but the results should be output 1 or 2
        results[i] = winner + 1
        i += 1
    return (weights1, weights2)

def main():
    df = importCSV('dataset_noclass.csv')
    
    weights1, weights2 = cluster(df)
    
    # percision = precision_score(expectedOutputList, actualOutputList, average='weighted')
    # recall = recall_score(expectedOutputList, actualOutputList, average='weighted')
    
    with open('output.txt', 'a') as outputFile:
        outputFile.write("\nFinal weights: ")
        outputFile.write(str(weights))
        # outputFile.write("\nPercision score: %.2f\n" % percision)
        # outputFile.write("\nRecall score: %.2f\n" % recall)
        # outputFile.write("\nConfusion matrix: \n%s" % confusion_matrix(expectedOutputList, actualOutputList))

main()