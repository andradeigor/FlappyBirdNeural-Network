from abc import ABC, abstractmethod
import numpy as np

class MutationStrategy(ABC):
    @abstractmethod
    def mutate(self, newPopulation, mutationRate, lowBound,highBound):
        pass

# Mutation: NewValue
class ClassicMutation:
    def mutate(self, newPopulation, mutationRate, lowBound,highBound):
        sigma = 0.1
        #For que percorre toda a camadas das rede neurais, indo em cada peso e bias
        for NNIndex in range(len(newPopulation)):
            for layerIndex in range(len(newPopulation[NNIndex].Layers)):

                for biasesIndex in range(len(newPopulation[NNIndex].Layers[layerIndex].biases)):
                    for item in range(len(newPopulation[NNIndex].Layers[layerIndex].biases[biasesIndex])):
                       # Verifica se ocorre uma mutação e aplica
                        if np.random.rand() < mutationRate:
                            newPopulation[NNIndex].Layers[layerIndex].biases[biasesIndex][item] = np.random.uniform(low=lowBound, high=highBound)

                for lineIndex in range(len(newPopulation[NNIndex].Layers[layerIndex].weights)):
                    for item in range(len(newPopulation[NNIndex].Layers[layerIndex].weights[lineIndex])):
                        # Verifica se ocorre uma mutação e aplica
                        if np.random.rand() < mutationRate:
                            newPopulation[NNIndex].Layers[layerIndex].biases[biasesIndex][item] =np.random.uniform(low=lowBound, high=highBound)

# Mutation: Gaussian
class GaussianMutation:
    def mutate(self, newPopulation, mutationRate, lowBound,highBound):
        sigma=0.1
        for NNIndex in range(len(newPopulation)):
            for layerIndex in range(len(newPopulation[NNIndex].Layers)):
                for biasesIndex in range(len(newPopulation[NNIndex].Layers[layerIndex].biases)):
                    for item in range(len(newPopulation[NNIndex].Layers[layerIndex].biases[biasesIndex])):
                        if np.random.rand() < mutationRate:
                            newPopulation[NNIndex].Layers[layerIndex].biases[biasesIndex][item] += np.random.normal(0, sigma)
                            newPopulation[NNIndex].Layers[layerIndex].biases[biasesIndex][item] = np.clip(
                                newPopulation[NNIndex].Layers[layerIndex].biases[biasesIndex][item],
                                lowBound, highBound
                            )
                for lineIndex in range(len(newPopulation[NNIndex].Layers[layerIndex].weights)):
                    for item in range(len(newPopulation[NNIndex].Layers[layerIndex].weights[lineIndex])):
                        if np.random.rand() < mutationRate:
                            newPopulation[NNIndex].Layers[layerIndex].weights[lineIndex][item] += np.random.normal(0, sigma)
                            newPopulation[NNIndex].Layers[layerIndex].weights[lineIndex][item] = np.clip(
                                newPopulation[NNIndex].Layers[layerIndex].weights[lineIndex][item],
                                lowBound, highBound
                            )
