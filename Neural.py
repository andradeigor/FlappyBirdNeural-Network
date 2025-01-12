import numpy as np

class NeuralNetwork:
    def __init__(self,shape=None):
        self.fitness =0
        Layers = []
        for i in range(1,len(shape)-1):#vai até o penultimo tamanho
            Layers.append(Layer(shape[i-1], shape[i]))
        Layers.append(Layer(shape[len(shape)-2], shape[-1]))
        self.Layers = Layers

    def printNN(self):
        for index,layer in enumerate(self.Layers):
            print(f'Printando a Layer {index}')
            layer.printLayer()
            print('-'* 50)

    def feedforward(self, input):
        currentInput = input
        for i in range(len(self.Layers)):
            currentLayer = self.Layers[i]
            currentLayer.foward(currentInput)
            currentLayer.tanh(currentLayer.output)
            currentInput = currentLayer.result
        return sum(currentInput)
        

class Layer:
    def __init__(self,nInput=None,nNeurons=None):
        if(nInput==None or nNeurons==None): return
        self.weights = np.array(0.5 * np.random.randn(nNeurons,nInput))
        self.biases = np.array(0.2 * np.random.randn(1,nNeurons))
    def foward(self, input):
        self.output = np.dot(input,self.weights.T) + self.biases

    def tanh(self,values):
        self.result = np.tanh(values)
    def printLayer(self):
        print(f"Layer Weights:")
        print(self.weights)
        print(f"Layer Biases:")
        print(self.biases)
