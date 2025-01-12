import numpy as np
from Neural import NeuralNetwork

def test_feedforward():
    # Configurar pesos e bias conhecidos para a camada
    class MockLayer:
        def __init__(self, weights, biases):
            self.weights = np.array(weights)
            self.biases = np.array(biases)

        def forward(self, input):
            self.output = (np.dot(input, self.weights.T) + self.biases)
        def tanh(self,values):
            self.result = np.tanh(values)

    # Substituir as camadas reais por camadas mock
    class TestNeuralNetwork(NeuralNetwork):
        def __init__(self):
            self.Layers = [
                MockLayer(weights=[[0.5, -0.2], [0.3, 0.8]], biases=[[0.1, -0.1]]),
                MockLayer(weights=[[0.7, -0.5]], biases=[[0.2]])
            ]

    # Criar a rede neural de teste
    network = TestNeuralNetwork()

    # Entrada conhecida
    input_data = np.array([[1.0, 2.0]])

    # Saída esperada
    # Camada 0: np.tanh(np.dot([[1.0, 2.0]], [[0.5, -0.2], [0.3, 0.8]].T) + [0.1, -0.1]) = [[0.19737532 0.94680601]]
    dot_layer_0 = np.dot(input_data, np.array([[0.5, -0.2], [0.3, 0.8]]).T) + np.array([[0.1, -0.1]])
    layer_0_output = np.tanh(dot_layer_0)
    print(f'O dot foi {dot_layer_0}\noutput da camada 0 foi {layer_0_output}')
    # Camada 1: np.tanh(np.dot([[0.19737532 0.94680601]], [[0.7, -0.5]].T) + [0.2]) = −0.134421757
    expected_output = np.tanh(np.dot(layer_0_output, np.array([[0.7, -0.5]]).T) + np.array([[0.2]]))
    print(f'O output foi {expected_output}')
    # Obter saída da rede
    output = network.feedforward(input_data)
    # Validar a saída
    assert np.allclose(output, expected_output, atol=1e-6), f"Saída inválida: {output} != {expected_output}"
    print("Teste de feedforward passou com sucesso!")

# Executar o teste
test_feedforward()
