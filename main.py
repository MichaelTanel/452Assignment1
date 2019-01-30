import csv
import pandas as pd
from random import uniform

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
    activationValue = 0
    for i in range(len(values)):
        activationValue += values[i] * weights[i + 1]

    return activationValue

# Calculating the new weights using the error correction learning technique
def calculateNewWeights(output, weights, values, expectedOutput):
    learningRate = 0.4
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
    return 1 if activation >= 0 else 0

def train():
    df = importCSV('trainSeeds.csv')

    global totalCount

    weights1 = []
    weights2 = []
    values = []

    weights1 = [uniform(-1, 1) for _ in range(8)]
    weights2 = [uniform(-1, 1) for _ in range(8)]
    
    print(weights1)
    print(weights2)

    for i in range(0, 300):    
        errorCount = 0
        totalCount = 0

        # TODO: add exit if success rate is too high

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

            # If either of the outputs did not match their corresponding bit in the expected output,
            # increase the error.
            if int(expectedOutputBinary[0]) != output1 and int(expectedOutputBinary[1]) != output2:
                errorCount += 1

            totalCount += 1

        print(totalCount - errorCount)
        print(errorCount)
        print(totalCount)
        print("Success rate: ", (float(totalCount) - float(errorCount)) / float(totalCount))

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
    errorTestCount = 0

    totalCount = 0

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

        totalCount += 1
    
    print("Success Count: ", successTestCount)
    print("Total Count: ", totalCount)    
    print("Success Rate: ", float(successTestCount) / float(totalCount))

def main():
    (weights1, weights2) = train()
    test(weights1, weights2)
    print(weights1)
    print("-----------")
    print(weights2)
main()