import csv
from pprint import pprint as pprint
import pandas as pd
import random

class Neuron(object):
    def __init__(self):
        self.area = 0
        self.weightArea = random.random()
        self.perimeter = 0
        self.weightPerimeter = random.random()
        self.compactness = 0
        self.weightCompactness = random.random()
        self.length = 0
        self.weightLength = random.random()
        self.width = 0
        self.weightWidth = random.random()
        self.asymmetryCoefficient = 0
        self.weightAsymmetryCoefficient = random.random()
        self.lengthGroove = 0
        self.weightLengthGroove = random.random()
        self.expectedOutput = 0
        self.output = ""

    # def __init__(self):
    #     self._weightArea = None

    # @weightArea.setter
    # def weightArea(self, value):
    #     self._weightArea = value

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
    # print(df)
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

def initializeNeuron(neuron):
    neuron.weightArea = random.random()
    neuron.weightPerimeter = random.random()
    neuron.weightCompactness = random.random()
    neuron.weightLength = random.random()
    neuron.weightWidth = random.random()
    neuron.weightAsymmetryCoefficient = random.random()
    neuron.weightLengthGroove = random.random()

    return neuron

def main():
    df = importCSV('trainSeeds.csv')

    neuron1 = Neuron()
    neuron2 = Neuron()

    print(neuron1.weightArea)
    print(neuron2.weightArea)

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
        neuron1.expectedOutput = row[1][7]
        
        neuron2.area = row[1][0]
        neuron2.perimeter = row[1][1]
        neuron2.compactness = row[1][2]
        neuron2.length = row[1][3]
        neuron2.width = row[1][4]
        neuron2.asymmetryCoefficient = row[1][5]
        neuron2.lengthGroove = row[1][6]
        neuron2.expectedOutput = row[1][7]

        # print(neuron1.area)
        # print(neuron2.area)
        # print('\n------------------------------------------------\n')
        
main()