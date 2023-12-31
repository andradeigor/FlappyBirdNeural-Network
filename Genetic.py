from Neural import NeuralNetwork, Layer 
from random import choice
import numpy as np

class Genetic:
    def __init__(self,population,mutationRate,shape,parentsNumber):
        self.totalWeights = 0
        self.shape = shape
        self.totalBiases = sum(shape[1:])
        self.mutationRate = mutationRate
        self.populationList = [NeuralNetwork(shape) for x in range(population)]
        self.parentsNumber = parentsNumber
        for i in range(1,len(shape)):
            self.totalWeights+= shape[i-1]*shape[i]
        


    def selection(self):
        populationOrderByFitness = sorted(self.populationList, key=lambda x: x.fitness, reverse=True)
  
        selected = []
        for i in range(self.parentsNumber):
            selected.append(populationOrderByFitness[i])

        return selected
    #Isso aqui ficou tão confuso que nem eu sei mais o que está rolando
    def crossOver(self,selected):
        newPopulation = [NeuralNetwork(self.shape) for x in range(len(self.populationList))]
        for populationIndex in range(len(self.populationList)):
            crossOverPointWeights = np.random.randint(0,self.totalWeights)
            crossOverPointBiases = np.random.randint(0,self.totalBiases)
            replacedWeights = 0
            replacedBiases = 0
            for layerIndex in range(len(selected[0].Layers)):
                currentLayer = selected[0].Layers[layerIndex] # Entende-se que as duas layers selecionadas vão ter o mesmo tamanho
                newLayer = Layer()
                newWeights = []
                newBiases = []
                for i in range(len(currentLayer.weights)):
                    newWightsLine = []
                    for j in range(len(currentLayer.weights[0])):
                        if(replacedWeights<crossOverPointWeights):
                            newWightsLine.append(selected[0].Layers[layerIndex].weights[i][j]) #
                        else:
                            newWightsLine.append(selected[1].Layers[layerIndex].weights[i][j])
                        replacedWeights+=1
                    newWeights.append(newWightsLine)
                for i in range(len(currentLayer.biases)):
                    if(replacedBiases < crossOverPointBiases):
                        newBiases.append(selected[0].Layers[layerIndex].biases[i])
                    else:
                        newBiases.append(selected[1].Layers[layerIndex].biases[i])
                    replacedBiases+=1
                newLayer.weights = np.array(newWeights)
                newLayer.biases = np.array(newBiases)
                newPopulation[populationIndex].Layers[layerIndex] = newLayer
        return newPopulation

    def mutate(self):
        for NNIndex in range(len(self.populationList)):
            for layerIndex in range(len(self.populationList[NNIndex].Layers)):
                for lineIndex in range(len(self.populationList[NNIndex].Layers[layerIndex].weights)):
                    for item in range(len(self.populationList[NNIndex].Layers[layerIndex].weights[lineIndex])):
                        if(np.random.rand()< self.mutationRate):
                            self.populationList[NNIndex].Layers[layerIndex].weights[lineIndex][item] = 0.2 * np.random.randn()

    def evolve(self):
        selected = self.selection()
        newPopulation = self.crossOver(selected)

        self.populationList = newPopulation

        self.mutate()







