import numpy as np

class NeuralNetwork:
    def __init__(self,shape=None):
        self.fitness =0
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
    def __init__(self,nInput=None,nNeurons=None):
        if(nInput==None or nNeurons==None): return
        self.weights = np.array(0.2 * np.random.randn(nNeurons,nInput))
        self.biases = np.array(0.2 * np.random.randn(1,nNeurons))
    def foward(self, input):
        self.output = np.dot(input,self.weights.T) + self.biases

    def reLU(self,values):
        self.result = np.maximum(0,values)
    def sigmoid(self,values):
        self.result = (1 / (1 + np.exp(-values)))


