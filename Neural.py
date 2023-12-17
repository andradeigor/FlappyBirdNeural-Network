import numpy as np


shape = [2,4,4,1] #4
class NeuralNetwork:
    def __init__(self,shape):
        Layers = []
        for i in range(1,len(shape)-1):#vai até o penultimo tamanho
            Layers.append(Layer(shape[i-1], shape[i]))
        Layers.append(Layer(shape[len(shape)-2], 1))
        self.Layers = Layers
    def printNN(self):
        for layer in self.Layers:
            print(layer.weights)

    def feedforward(self, input):
        currentInput = input
        for i in range(len(self.Layers)):
            currentLayer = self.Layers[i]
            currentLayer.foward(currentInput)

            if(i!= (len(self.Layers)-1)):
                currentLayer.reLU(currentLayer.output)
            else:
                currentLayer.sigmoid(currentLayer.output)

            currentInput = currentLayer.result
        return sum(currentInput)
        

class Layer:
    def __init__(self,nInput,nNeurons):
        self.weights = 0.2 * np.random.randn(nInput,nNeurons)
        self.biases = 0.2 * np.random.randn(1,nNeurons)
    def foward(self, input):
        self.output = np.dot(input,self.weights) + self.biases

    def reLU(self,values):
        self.result = np.maximum(0,values)
    def sigmoid(self,values):
        self.result = (1 / (1 + np.exp(-values)))



NN = NeuralNetwork([2,4,4,1])
X = [500,350]
print((NN.feedforward(X)))