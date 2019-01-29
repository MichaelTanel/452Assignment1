import csv
from pprint import pprint as pprint
import pandas as pd

class Neuron(object):
    area = 0
    weigthArea = 0.5
    perimeter = 0
    weightPerimeter = 0.5
    compactness = 0
    weightCompactness = 0.5
    length = 0
    weightLength = 0.5
    width = 0
    weightWidth = 0.5
    asymmetryCoefficient = 0
    weightAsymmetryCoefficient = 0.5
    lengthGroove = 0
    weightLengthGroove = 0.5
    expectedOutput = 0
    output = ""

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
    maxVals.area = max(df['Area']) #you can also use df['column_name']
    maxVals.perimeter = max(df['Perimeter'])
    maxVals.compactness = max(df['Compactness'])
    maxVals.length = max(df['Length'])
    maxVals.width = max(df['Width'])
    maxVals.asymmetryCoefficient = max(df['AsymmetryCoefficient'])
    maxVals.lengthGroove = max(df['LengthGroove'])

    return normalizeData(maxVals, df)

def main():
    df = importCSV('trainSeeds.csv')

    # Iterate over dataframe rows
    for row in df.iterrows():
        # In order to get the data point, must access at [1][n], 0 < n < 8
        # This is because row[0] gives the row number, row[1] gives the
        # column name and the data point
        neuron = Neuron()
        neuron.area = row[1][0]
        neuron.perimeter = row[1][1]
        neuron.compactness = row[1][2]
        neuron.length = row[1][3]
        neuron.width = row[1][4]
        neuron.asymmetryCoefficient = row[1][5]
        neuron.lengthGroove = row[1][6]
        neuron.expectedOutput = row[1][7]
        print('\n------------------------------------------------\n')
        
main()