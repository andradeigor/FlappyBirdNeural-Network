import numpy as np

class NeuralNetwork:
    def __init__(self,shape,lowBound, highBound):
        self.fitness = 0
        self.Layers = [
            Layer(shape[i], shape[i + 1], lowBound, highBound) 
            for i in range(len(shape) - 1)
        ]

    def summary(self):
        for index,layer in enumerate(self.Layers):
            print(f'Printando a Layer {index}')
            layer.printLayer()
            print('-'* 50)

    def feedforward(self, input):
        currentInput = input
        for i in range(len(self.Layers)):
            currentLayer = self.Layers[i]
            currentLayer.forward(currentInput)
            currentLayer.tanh(currentLayer.output)
            currentInput = currentLayer.result
        #print(currentInput)
        return sum(currentInput)
        

class Layer:
    def __init__(self,nInput,nNeurons,lowBound,highBound):
        self.weights = np.random.uniform(low=lowBound, high=highBound, size=(nNeurons, nInput))
        self.biases = np.random.uniform(low=lowBound, high=highBound, size=(1, nNeurons))
        self.output = None
    def forward(self, input):
        self.output = np.dot(input, self.weights.T) + self.biases

    def tanh(self,values):
        self.result = np.tanh(values)
    
    def printLayer(self):
        print(f"Layer Weights:")
        print(self.weights)
        print(f"Layer Biases:")
        print(self.biases)


