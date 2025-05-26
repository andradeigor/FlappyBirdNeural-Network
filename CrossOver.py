from abc import ABC, abstractmethod
from Neural import NeuralNetwork
import numpy as np

class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1, parent2, shape, lowBound, highBound, alpha):
        pass



# Crossover: Binário
class BinaryCrossOver(CrossoverStrategy):
    def crossover(self, parent1, parent2, shape, lowBound, highBound, alpha):
        son = NeuralNetwork(shape, lowBound, highBound)
        for layerIndex in range(len(parent1.Layers)):
            for biasesIndex in range(len(parent1.Layers[layerIndex].biases)):
                for item in range(len(parent1.Layers[layerIndex].biases[biasesIndex])):
                    genSelected = parent1 if np.random.rand() > 0.5 else parent2
                    son.Layers[layerIndex].biases[biasesIndex][item] = genSelected.Layers[layerIndex].biases[biasesIndex][item]
            for weightIndex in range(len(parent1.Layers[layerIndex].weights)):
                for item in range(len(parent1.Layers[layerIndex].weights[weightIndex])):
                    genSelected = parent1 if np.random.rand() > 0.5 else parent2
                    son.Layers[layerIndex].weights[weightIndex][item] = genSelected.Layers[layerIndex].weights[weightIndex][item]
        return son
    

# Crossover: Mean
class MeanCrossOver(CrossoverStrategy):
    def crossover(self, parent1, parent2, shape, lowBound, highBound, alpha):
        son = NeuralNetwork(self.shape,self.lowBound, self.highBound) 
        #For que percorre toda a camadas das rede neurais, indo em cada peso e bias
        for layerIndex in range(len(parent1.Layers)):
            for biasesIndex in range(len(parent1.Layers[layerIndex].biases)):
                for item in range(len(parent1.Layers[layerIndex].biases[biasesIndex])):
                    #Seleciona pai1 ou dois bináriamente e atribui 
                    genParent1 = parent1.Layers[layerIndex].biases[biasesIndex][item]
                    genParent2 = parent2.Layers[layerIndex].biases[biasesIndex][item]
                    son.Layers[layerIndex].biases[biasesIndex][item] = (genParent1 + genParent2)/2
                
            for weightIndex in range(len(parent1.Layers[layerIndex].weights)):
                for item in range(len(parent1.Layers[layerIndex].weights[weightIndex])):
                    genParent1 = parent1.Layers[layerIndex].weights[weightIndex][item]
                    genParent2 = parent2.Layers[layerIndex].weights[weightIndex][item]
                    son.Layers[layerIndex].weights[weightIndex][item] = (genParent1 + genParent2)/2

        return son

# Crossover: Geométrico
class GeometricCrossOver(CrossoverStrategy):
    def crossover(self, parent1, parent2, shape, lowBound, highBound, alpha):
        son = NeuralNetwork(self.shape,self.lowBound, self.highBound) 
        #For que percorre toda a camadas das rede neurais, indo em cada peso e bias
        for layerIndex in range(len(parent1.Layers)):
            for biasesIndex in range(len(parent1.Layers[layerIndex].biases)):
                for item in range(len(parent1.Layers[layerIndex].biases[biasesIndex])):
                    #Seleciona pai1 ou dois bináriamente e atribui 
                    genParent1 = parent1.Layers[layerIndex].biases[biasesIndex][item]
                    genParent2 = parent2.Layers[layerIndex].biases[biasesIndex][item]
                    son.Layers[layerIndex].biases[biasesIndex][item] = (genParent1 * genParent2)**2
                
            for weightIndex in range(len(parent1.Layers[layerIndex].weights)):
                for item in range(len(parent1.Layers[layerIndex].weights[weightIndex])):
                    genParent1 = parent1.Layers[layerIndex].weights[weightIndex][item]
                    genParent2 = parent2.Layers[layerIndex].weights[weightIndex][item]
                    son.Layers[layerIndex].weights[weightIndex][item] = (genParent1 * genParent2)**2

        return son


# Crossover: BLX-Alpha
class CrossOverBLX(CrossoverStrategy):
    def crossover(self, parent1, parent2, shape, lowBound, highBound, alpha):
        son = NeuralNetwork(shape, lowBound, highBound)
        for layerIndex in range(len(parent1.Layers)):
            for biasesIndex in range(len(parent1.Layers[layerIndex].biases)):
                for item in range(len(parent1.Layers[layerIndex].biases[biasesIndex])):
                    genParent1 = parent1.Layers[layerIndex].biases[biasesIndex][item]
                    genParent2 = parent2.Layers[layerIndex].biases[biasesIndex][item]
                    beta = np.random.uniform(-alpha, 1 + alpha)
                    son.Layers[layerIndex].biases[biasesIndex][item] = genParent1 + beta * (genParent2 - genParent1)
            for weightIndex in range(len(parent1.Layers[layerIndex].weights)):
                for item in range(len(parent1.Layers[layerIndex].weights[weightIndex])):
                    genParent1 = parent1.Layers[layerIndex].weights[weightIndex][item]
                    genParent2 = parent2.Layers[layerIndex].weights[weightIndex][item]
                    beta = np.random.uniform(-alpha, 1 + alpha)
                    son.Layers[layerIndex].weights[weightIndex][item] = genParent1 + beta * (genParent2 - genParent1)
        return son