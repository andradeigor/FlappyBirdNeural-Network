import numpy as np

# Inicialização de pesos com He, apropriada para ReLU
def init_weights_he(nInput, nNeurons):
    stddev = np.sqrt(2 / nInput)
    return np.random.normal(0, stddev, size=(nNeurons, nInput))

# Inicialização de pesos com Xavier, apropriada para tanh/sigmoid
def init_weights_xavier(nInput, nNeurons):
    limit = np.sqrt(6 / (nInput + nNeurons))
    return np.random.uniform(-limit, limit, size=(nNeurons, nInput))


class NeuralNetwork:
    def __init__(self,shape,lowBound, highBound):
        self.fitness = 0
        self.Layers = [
            Layer(shape[i], shape[i + 1], lowBound, highBound) 
            for i in range(len(shape) - 1)
        ]
    #Apenas para debug
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
            
            if(i == (len(self.Layers)-1)):
                currentLayer.softmax(currentLayer.output)  # Última camada: softmax
            else:
                #currentLayer.tanh(currentLayer.output)
                currentLayer.relu(currentLayer.output)     # Ocultas: ativação selecionada

            currentInput = currentLayer.result
        return currentInput
        

class Layer:
    def __init__(self, nInput, nNeurons, lowBound, highBound):
        #inicialização uniforme dos pesos e biases. As vezes testo com os métodos implementados anteriores, dizem ser melhor para RELU/TANH
        self.weights = np.random.uniform(low=lowBound, high=highBound, size=(nNeurons, nInput))
        self.biases = np.random.uniform(low=lowBound, high=highBound, size=(1, nNeurons))
        #self.weights = init_weights_he(nInput, nNeurons)
        #self.biases = init_weights_he(1, nNeurons)
        self.output = None
    def forward(self, input):
        self.output = np.dot(input, self.weights.T) + self.biases

    def tanh(self,values):
        self.result = np.tanh(values)
    
    def relu(self, values):
        self.result = np.maximum(0, values)
    def softmax(self,values):
        exp_values = np.exp(values - np.max(values)) #Evita overflow
        self.result = exp_values / np.sum(exp_values)
        #print(self.result)
    
    def printLayer(self):
        print(f"Layer Weights:")
        print(self.weights)
        print(f"Layer Biases:")
        print(self.biases)


