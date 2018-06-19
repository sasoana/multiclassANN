from numpy.core.tests.test_mem_overlap import xrange

from neuralnetwork import *

# X = (amino acid), y = secondary structure
# amino acid encoding
encoding = {
    "A": 0.05, "C": 0.1, "E": 0.15, "D": 0.2, "G": 0.25, "F": 0.3, "I": 0.35, "H": 0.4, "K": 0.45, "M": 0.5,
    "L": 0.55, "N": 0.6, "Q": 0.65, "P": 0.7, "S": 0.75, "R": 0.8, "T": 0.85, "W": 0.9, "V": 0.95, "Y": 1
}

trainFile = open("protein-secondary-structure.train.txt","r")
proteins = []
secondaryStructures = []
data = trainFile.read().splitlines(keepends=False)
protein = []
secondaryStructure = []
for line in data:
    tokens = line.split()
    if len(tokens) == 0:
        continue
    if tokens[0] == "end":
        proteins.append(protein)
        secondaryStructures.append(secondaryStructure)
        continue
    if tokens[0] == "<>":
        protein = []
        secondaryStructure = []
    else:
        protein.append(encoding[tokens[0]])
        if tokens[1] == "h":
            actualOutput = [1, 0, 0]
        if tokens[1] == "e":
            actualOutput = [0, 1, 0]
        if tokens[1] == "_":
            actualOutput = [0, 0, 1]
        secondaryStructure.append(actualOutput)
trainFile.close()
step = 13 // 2
start = 13 // 2
X = np.array(([proteins[0][start-step:start+step+1]]), dtype=float)
# take sequences of 13 amino acids from each protein
for protein in proteins:
    seqLength = len(protein)
    for i in xrange(start+1, seqLength-step):
        X = np.concatenate((X, np.array(([protein[i-step:i+step+1]]), dtype=float)))
#print(X[:50])

start = 13 // 2
y = np.array(([secondaryStructures[0][start]]), dtype=float)
for structure in secondaryStructures:
    seqLength = len(structure)
    for i in xrange(start+1, seqLength-step):
        y = np.concatenate((y, np.array(([structure[i]]), dtype=float)))
#print(y[:15])

testFile = open("protein-secondary-structure.test.txt","r")
proteinsTest = []
secondaryStructuresTest = []
data = testFile.read().splitlines(keepends=False)
protein = []
secondaryStructure = []
for line in data:
    tokens = line.split()
    if len(tokens) == 0:
        continue
    if tokens[0] == "end":
        proteinsTest.append(protein)
        secondaryStructuresTest.append(secondaryStructure)
        continue
    if tokens[0] == "<>":
        protein = []
        secondaryStructure = []
    else:
        protein.append(encoding[tokens[0]])
        if tokens[1] == "h":
            actualOutput = [0]
        if tokens[1] == "e":
            actualOutput = [1]
        if tokens[1] == "_":
            actualOutput = [2]
        secondaryStructure.append(actualOutput)
testFile.close()
step = 13 // 2
start = 13 // 2
xPredicted = np.array(([proteinsTest[0][start-step:start+step+1]]), dtype=float)
# take sequences of 13 amino acids from each protein
for protein in proteinsTest:
    seqLength = len(protein)
    for i in xrange(start+1, seqLength-step):
        xPredicted = np.concatenate((xPredicted, np.array(([protein[i-step:i+step+1]]), dtype=float)))
#print(X[:50])

start = 13 // 2
yActual = np.array(([secondaryStructuresTest[0][start]]), dtype=float)
for structure in secondaryStructuresTest:
    seqLength = len(structure)
    for i in xrange(start+1, seqLength-step):
        yActual = np.concatenate((yActual, np.array(([structure[i]]), dtype=float)))

nn = NeuralNetwork.init(0.03, 13, 3, [10])
model = nn.train(X, y)
prediction = model.predict_multiclass_classification(xPredicted)
print(prediction[:50])
allMatches = 0
for i in xrange(len(xPredicted)):
    if prediction[i] == yActual[i]:
        allMatches += 1
print("Q3: " + str(allMatches/len(xPredicted) * 100))

