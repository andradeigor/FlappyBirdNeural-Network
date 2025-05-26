from Genetic import Genetic
import copy
import numpy as np
from Selection import SelectionByTournament
from CrossOver import CrossOverBLX
from Mutations import GaussianMutation
from Neural import NeuralNetwork
def evaluate_network(nn, X, y):
    predictions = []
    for input_data, target in zip(X, y):
        output = nn.feedforward(input_data.reshape(1, -1))
        predictions.append(output)
    predictions = np.array(predictions).reshape(-1, 1)
    loss = np.mean((predictions - y) ** 2)  # Erro quadrático médio (MSE)
    fitness = 1 / (loss + 1e-6)
        
    return fitness

def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Entradas
    y = np.array([[0], [1], [1], [0]])  # Saídas esperadas
    #np.random.seed(42)
    # Configuração da rede neural
    shape = [2, 4, 1]  # 2 entradas, 4 neurônios na camada oculta, 1 saída
    lowBound = -1
    highBound = 1.0
    g = Genetic(30,0.05,shape,15,lowBound,highBound,True,selection_strategy=SelectionByTournament,crossover_strategy=CrossOverBLX,mutation_strategy=GaussianMutation)
    meanLost = 0
    """
    best_Score = float('-inf')
    best_NN = None
    for i in range(100):
        meanLost = 0
        for nn in g.populationList: 
            nn.fitness = evaluate_network(nn, X, y)
            meanLost += nn.fitness
            if(nn.fitness> best_Score):
                best_Score= nn.fitness
                best_NN = copy.copy(nn)
        meanLost/=30
        g.evolve()

    print(f'O melhor score foi: {best_Score}, estamos esperando ')
    print(y)
    for input_data in X:
        print('[',end='')
        print(f'{best_NN.feedforward(input_data.reshape(1, -1))} ',end='')
    print(']')
    """
    rede = NeuralNetwork(shape,-1.0,1.0)
    rede.train(X,y,100000,0.01)
    print(rede.feedforward(X))


if __name__ == "__main__":
    main()

