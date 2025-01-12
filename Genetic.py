from Neural import NeuralNetwork, Layer 
from random import choices
import numpy as np

class Genetic:
    def __init__(self,population,mutationRate,shape,parentsNumber):
        self.shape = shape
        self.mutationRate = mutationRate
        self.populationList = [NeuralNetwork(shape) for x in range(population)]
        self.parentsNumber = parentsNumber
        


    def selection(self):
        populationScore = np.array([i.fitness for i in self.populationList])
        populationProbability = populationScore / sum(populationScore)
        selectedIndexes = np.random.choice(len(populationProbability), size=self.parentsNumber, p=populationProbability, replace=False)
        selected = [ self.populationList[i] for i in selectedIndexes]
        return selected


    def crossOver(self, selected):
        newPopulation = [NeuralNetwork(self.shape) for x in range(len(self.populationList))]
        for NNIndex in range(len(self.populationList)):
            for layerIndex in range(len(self.populationList[NNIndex].Layers)):
                for biasesIndex in range(len(self.populationList[NNIndex].Layers[layerIndex].biases)):
                    for item in range(len(self.populationList[NNIndex].Layers[layerIndex].biases[biasesIndex])):
                        baseIndex = np.random.randint(0,self.parentsNumber+1)
                        newPopulation[NNIndex].Layers[layerIndex].biases[biasesIndex][item] = selected[baseIndex].Layers[layerIndex].biases[biasesIndex][item]
                    

                for weightIndex in range(len(self.populationList[NNIndex].Layers[layerIndex].weights)):
                    for item in range(len(self.populationList[NNIndex].Layers[layerIndex].weights[weightIndex])):
                        baseIndex = np.random.randint(0,self.parentsNumber+1)
                        
                        newPopulation[NNIndex].Layers[layerIndex].weights[weightIndex][item] = selected[baseIndex].Layers[layerIndex].weights[weightIndex][item]
                        
        return newPopulation

    def mutate(self):
        for NNIndex in range(len(self.populationList)):
            for layerIndex in range(len(self.populationList[NNIndex].Layers)):

                for biasesIndex in range(len(self.populationList[NNIndex].Layers[layerIndex].biases)):
                    for item in range(len(self.populationList[NNIndex].Layers[layerIndex].biases[biasesIndex])):
                        if(np.random.rand()< self.mutationRate):
                            self.populationList[NNIndex].Layers[layerIndex].biases[biasesIndex][item] =  np.random.uniform(low=-30,high=30)



                for lineIndex in range(len(self.populationList[NNIndex].Layers[layerIndex].weights)):
                    for item in range(len(self.populationList[NNIndex].Layers[layerIndex].weights[lineIndex])):
                        if(np.random.rand()< self.mutationRate):
                            self.populationList[NNIndex].Layers[layerIndex].weights[lineIndex][item] =  np.random.uniform(low=-30,high=30)

    def evolve(self):
        selected = self.selection()
        newPopulation = self.crossOver(selected)

        self.populationList = newPopulation

        self.mutate()










