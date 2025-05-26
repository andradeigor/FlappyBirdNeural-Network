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


    def compute_loss(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred + 1e-8))  # Cross-Entropy


    def backpropagate(self, X, y_true, learning_rate):
        y_pred = self.feedforward(X)

        delta =y_pred - y_true
        #Percorrendo a rede de trás pare frente
        for i in reversed(range(len(self.Layers))):
            layer = self.Layers[i]

            # dW = delta.T * input
            dW = np.dot(delta.T, layer.input)
            db = np.sum(delta, axis=0, keepdims=True)

            max_norm = 5.0
            if np.linalg.norm(dW) > max_norm:
                dW = dW * (max_norm / np.linalg.norm(dW))
            # Atualiza parâmetros
            layer.weights -= learning_rate * dW
            layer.biases -= learning_rate * db


            if i != 0:
                prev_layer = self.Layers[i-1]
                delta = np.dot(delta, layer.weights)

                delta *= prev_layer.relu_derivative()


    def feedforward(self, input):
        currentInput = input
        for i in range(len(self.Layers)):
            currentLayer = self.Layers[i]
            currentLayer.forward(currentInput)

            if(i == (len(self.Layers)-1)):
              currentLayer.softmax(currentLayer.output)  # Última camada: softmax
            else:
                currentLayer.relu(currentLayer.output)     # Ocultas: ativação selecionada

            currentInput = currentLayer.result
        return currentInput

    def train(self, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            loss = 0
            for i in range(0, len(X)):
                X_batch = X[i]
                # Garanta que X_batch seja (1, num_features)
                if X_batch.ndim == 1:
                    X_batch = X_batch.reshape(1, -1)

                Y_batch = Y[i]
                # Garanta que Y_batch seja (1, num_classes)
                # Se Y_batch já for one-hot e 1D (shape (num_classes,)), converta para 2D
                if Y_batch.ndim == 1:
                    Y_batch = Y_batch.reshape(1, -1)
                
                self.backpropagate(X_batch, Y_batch, learning_rate)
                y_pred = self.feedforward(X_batch) # X_batch já é 2D aqui
                loss += self.compute_loss(y_pred, Y_batch)

            if epoch % 10 == 0:
                # y_pred aqui é para o último X_batch do epoch
                print(y_pred) 
                print(f"Epoch {epoch}, Loss: {loss / len(X)}")


class Layer:
    def __init__(self, nInput, nNeurons, lowBound, highBound):
        #inicialização uniforme dos pesos e biases. As vezes testo com os métodos implementados anteriores, dizem ser melhor para RELU/TANH
        #self.weights = np.random.uniform(low=lowBound, high=highBound, size=(nNeurons, nInput))
        self.biases = np.full((1, nNeurons), 0.01)
        self.weights = np.random.randn(nNeurons, nInput) * np.sqrt(2 / nInput)
        #self.biases = init_weights_he(1, nNeurons)
        self.output = None
        self.input = None

    def relu_derivative(self):
        return (self.result > 0).astype(float)

    def tanh_derivative(self):
        return 1 - np.square(self.result)


    def forward(self, input):
        self.input =  np.array(input)
        self.output = np.dot(input, self.weights.T) + self.biases

    def tanh(self,values):
        self.result = np.tanh(values)

    def relu(self, values):
        self.result = np.maximum(0, values)
    def softmax(self,values):
        exp_values = np.exp(values - np.max(values)) #Evita overflow
        self.result = exp_values / np.sum(exp_values)
        #print("==================")
        #print(self.result)

    def printLayer(self):
        print(f"Layer Weights:")
        print(self.weights)
        print(f"Layer Biases:")
        print(self.biases)


