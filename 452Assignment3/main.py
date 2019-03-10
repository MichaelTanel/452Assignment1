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
    values.append(float(row[1][0]))
    values.append(float(row[1][1]))
    values.append(float(row[1][2]))

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

def calcOutput(inputValues, weights1, weights2):
    return (0, 1)

def cluster(df):

    global numInputNodes
    global numOutputNodes

    # Creates two 1D arrays with floating point numbers between -1 and 1.
    # weights1 represents the weights for the connections from all input nodes to output node 1.
    # weights2 represents the weights for the connections from all input nodes to output node 2.
    weights1 = [uniform(-1, 1) for x in range(numInputNodes)]
    weights2 = [uniform(-1, 1) for x in range(numInputNodes)]

    # Stores each row of the csv
    values = []
    
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

    # Iterate over dataframe row
    for row in df.iterrows():
        inputValues = parseRow(row)

        output1, output2 = calcOutput(inputValues, weights1, weights2)

        # While both outputs are not 0, keep updating values
        while output1 != 0 and output2 != 0:
            # Update connection weight
            num = 0
    
    return 0

def main():
    df = importCSV('dataset_noclass.csv')
    
    weights = cluster(df)
        
    # percision = precision_score(expectedOutputList, actualOutputList, average='weighted')
    # recall = recall_score(expectedOutputList, actualOutputList, average='weighted')
    
    with open('output.txt', 'a') as outputFile:
        outputFile.write("\nFinal weights: ")
        outputFile.write(str(weights))
        # outputFile.write("\nPercision score: %.2f\n" % percision)
        # outputFile.write("\nRecall score: %.2f\n" % recall)
        # outputFile.write("\nConfusion matrix: \n%s" % confusion_matrix(expectedOutputList, actualOutputList))

main()