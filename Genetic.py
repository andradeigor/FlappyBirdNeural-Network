from random import choices, sample
import numpy as np
from Neural import NeuralNetwork
class Genetic:
    def __init__(self, population, mutationRate, shape, parentsNumber, lowBound, highBound, elitism=False,p=0.75):
        #força a população a ser par
        population = population if population%2==0 else population+1
        self.shape = shape
        self.lowBound = lowBound
        self.highBound = highBound
        self.mutationRate = mutationRate
        self.populationList = [NeuralNetwork(shape, lowBound, highBound) for _ in range(population)]
        self.parentsNumber = parentsNumber
        self.elitism = elitism 
        self.p =p

    def selectionByTournament(self):
        selected = []
        for _ in range(self.parentsNumber):
            #Seleciona dois indivuduos aleatoriamente >COM< reposição, 
            # ou seja, um mesmo individuo pode ser selcionado mais de uma vez
            parent1, parent2 = sample(self.populationList,2)
            if parent1.fitness > parent2.fitness:
                best, worst = parent1, parent2
            else:
                best, worst = parent2, parent1
            #Pega o melhor com probabilidade p
            if(np.random.rand()<self.p):
                selected.append(best)
            else:
                selected.append(worst)
        return selected


    def selectionByRoulette(self):
        populationScore = np.array([i.fitness for i in self.populationList])

        # Prevenir divisão por zero
        if populationScore.sum() == 0:
            populationProbability = np.ones(len(populationScore)) / len(populationScore)
        else:
            populationProbability = populationScore / populationScore.sum()
        #escolhe os individuos com probabilidade proporcional ao seu fitness
        selectedIndexes = np.random.choice(len(populationProbability), size=self.parentsNumber, p=populationProbability, replace=True)
        selected = [self.populationList[i] for i in selectedIndexes]
        return selected


    def BinaryCrossOver(self,selected):
        newPopulation = []
        #For que roda na metade da população, cada vez criando 2 filhos. Assume que a população inteira é par
        for _ in range(len(self.populationList)//2):
            #Seleciona, com reposição, 2 pais
            parent1, parent2 = sample(selected, 2)
            son1 = self.crossOver(parent1,parent2)
            son2 = self.crossOver(parent1,parent2)
            #adiciona os dois novos filhos
            newPopulation.append(son1)
            newPopulation.append(son2)
        return newPopulation


    def crossOver(self, parent1, parent2):

        son = NeuralNetwork(self.shape,self.lowBound, self.highBound) 
        #For que percorre toda a camadas das rede neurais, indo em cada peso e bias
        for layerIndex in range(len(parent1.Layers)):
            for biasesIndex in range(len(parent1.Layers[layerIndex].biases)):
                for item in range(len(parent1.Layers[layerIndex].biases[biasesIndex])):
                    #Seleciona pai1 ou dois bináriamente e atribui 
                    genSelected = parent1 if np.random.rand()> 0.5 else parent2
                    son.Layers[layerIndex].biases[biasesIndex][item] = genSelected.Layers[layerIndex].biases[biasesIndex][item]
                
            for weightIndex in range(len(parent1.Layers[layerIndex].weights)):
                for item in range(len(parent1.Layers[layerIndex].weights[weightIndex])):
                    genSelected = parent1 if np.random.rand()> 0.5 else parent2
                    son.Layers[layerIndex].weights[weightIndex][item] = genSelected.Layers[layerIndex].weights[weightIndex][item]

        return son

    def mutate(self, newPopulation):
        #For que percorre toda a camadas das rede neurais, indo em cada peso e bias
        for NNIndex in range(len(newPopulation)):
            for layerIndex in range(len(newPopulation[NNIndex].Layers)):

                for biasesIndex in range(len(newPopulation[NNIndex].Layers[layerIndex].biases)):
                    for item in range(len(newPopulation[NNIndex].Layers[layerIndex].biases[biasesIndex])):
                        # Verifica se ocorre uma mutação e aplica
                        if np.random.rand() < self.mutationRate:
                            newPopulation[NNIndex].Layers[layerIndex].biases[biasesIndex][item] = np.random.uniform(low=self.lowBound, high=self.highBound)

                for lineIndex in range(len(newPopulation[NNIndex].Layers[layerIndex].weights)):
                    for item in range(len(newPopulation[NNIndex].Layers[layerIndex].weights[lineIndex])):
                        # Verifica se ocorre uma mutação e aplica
                        if np.random.rand() < self.mutationRate:
                            newPopulation[NNIndex].Layers[layerIndex].weights[lineIndex][item] = np.random.uniform(low=self.lowBound, high=self.highBound)

    def evolve(self):
        selected = self.selectionByTournament()
        newPopulation = self.BinaryCrossOver(selected)

        self.mutate(newPopulation)
        #se elitismo é verdade, pega o melhor indivíduo e passa para frente
        if self.elitism:
            best_individual = max(self.populationList, key=lambda nn: nn.fitness)
            newPopulation[0] = best_individual 

        self.populationList = newPopulation
