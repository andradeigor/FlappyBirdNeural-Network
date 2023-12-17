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
        
    
    def evaluete(self, inputs):
        for NN in self.populationList:
            for input in inputs:
                prediction = NN.feedforward(input)
                if(prediction>0.5):
                    if(input[0] > 200):
                        NN.fitness+=1
                    if(input[1] < 200):
                        NN.fitness+=1
                    if(input[0]>200 and input[1] < 200):
                        NN.fitness+=5
                else:
                    if(input[0] < 200):
                        NN.fitness+=1
                    if(input[1] > 200):
                        NN.fitness+=1
                    if(input[0]<200 and input[1] > 200):
                        NN.fitness+=5

    def selection(self):
        populationOrderByFitness = sorted(self.populationList, key=lambda x: x.fitness, reverse=True)
        totalFitness = sum(x.fitness for x in populationOrderByFitness)
        probabilityDistribution = [populationOrderByFitness[0].fitness/totalFitness]
        for i in range(1,len(populationOrderByFitness)):
            probabilityDistribution.append(probabilityDistribution[i-1] + populationOrderByFitness[i].fitness/totalFitness)

        selected = []
        for _ in range(self.parentsNumber):
            numberSelected = np.random.rand()
            for i,probability in enumerate(probabilityDistribution):
                if(numberSelected<=probability):
                    if(populationOrderByFitness[i] not in selected):
                        selected.append(populationOrderByFitness[i])
                    else:
                        
                        selected.append(choice(populationOrderByFitness))
                    break

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

    def run(self,inputs):
        for i in range(30):
            self.evaluete(inputs)
            #for NN in self.populationList:
                #for layer in NN.Layers:
                    #print()
            print(f"a média de fitness na geração {i+1} foi de:{ (sum(x.fitness for x in self.populationList))/len(self.populationList)}")
            selected = self.selection()
            newPopulation = self.crossOver(selected)

            self.populationList = newPopulation

            self.mutate()





G = Genetic(10,0.1,[2,4,4,1],2)
inputs = []
for i in range(1000):
    input = [np.random.randint(100,300),np.random.randint(100,300)]
    inputs.append(input)
G.run(inputs)


