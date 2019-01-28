import csv
from pprint import pprint as pprint

class Neuron(object):
    area = 0
    weigthArea = 0
    perimeter = 0
    weightPerimeter = 0
    compactness = 0
    weightCompactness = 0
    length = 0
    weightLength = 0
    width = 0
    weightWidth = 0
    asymmetryCoefficient = 0
    weightAsymmetryCoefficient = 0
    lengthGroove = 0
    weightLengthGroove = 0
    output = ""

def importCSV(filename):
    with open (filename, 'r') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        maxArea = 0
        maxPerimeter = 0
        maxCompactness = 0
        maxLength = 0
        maxWidth = 0
        maxAsymmetryCoefficient = 0
        maxLengthGroove = 0
        for row in rows:
            # use the float() method to get rid of strings
            neuron = Neuron()
            
            neuron.area = float(row[0])
            if neuron.area > maxArea:
                maxArea = neuron.area

            neuron.perimeter = float(row[1])            
            if neuron.perimeter > maxPerimeter:
                maxPerimeter = neuron.perimeter

            neuron.compactness = float(row[2])
            if neuron.compactness > maxCompactness:
                maxCompactness = neuron.compactness

            neuron.length = float(row[3])            
            if neuron.length > maxLength:
                maxLength = neuron.length

            neuron.width = float(row[4])            
            if neuron.width > maxWidth:
                maxWidth = neuron.width

            neuron.asymmetryCoefficient = float(row[5])
            if neuron.asymmetryCoefficient > maxAsymmetryCoefficient:
                maxAsymmetryCoefficient = neuron.asymmetryCoefficient

            neuron.lengthGroove = float(row[6])
            if neuron.lengthGroove > maxLengthGroove:
                maxLengthGroove = neuron.lengthGroove
        print(maxLength)


importCSV('testSeeds.csv')