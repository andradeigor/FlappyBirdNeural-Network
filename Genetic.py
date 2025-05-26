from random import  sample
from Neural import NeuralNetwork
from Selection import SelectionByTournament
from CrossOver import BinaryCrossOver
from Mutations import GaussianMutation

class Genetic:
    def __init__(self, population, mutationRate, shape, parentsNumber, lowBound, highBound,
                 elitism=False, p=0.75, alpha=0.3,
                 selection_strategy=None, crossover_strategy=None, mutation_strategy=None):
        
        population = population if population % 2 == 0 else population + 1
        self.shape = shape
        self.lowBound = lowBound
        self.highBound = highBound
        self.mutationRate = mutationRate
        self.populationList = [NeuralNetwork(shape, lowBound, highBound) for _ in range(population)]
        self.parentsNumber = parentsNumber
        self.elitism = elitism
        self.p = p
        self.alpha = alpha
        self.selection_strategy = selection_strategy or SelectionByTournament()
        self.crossover_strategy = crossover_strategy or BinaryCrossOver()
        self.mutation_strategy = mutation_strategy or GaussianMutation()



    def selectParentsAndCrossover(self,selected):
        newPopulation = []
        #For que roda na metade da população, cada vez criando 2 filhos. Assume que a população inteira é par
        for _ in range(len(self.populationList)//2):
            #Seleciona, com reposição, 2 pais
            parent1, parent2 = sample(selected, 2)
            son1 = self.crossover_strategy.crossover(self,parent1,parent2, self.shape, self.lowBound, self.highBound, self.alpha)
            son2 = self.crossover_strategy.crossover(self,parent1,parent2, self.shape, self.lowBound, self.highBound, self.alpha)
            #adiciona os dois novos filhos
            newPopulation.append(son1)
            newPopulation.append(son2)
        return newPopulation


    
  
    


    def evolve(self):
        selected = self.selection_strategy.select(self,self.populationList, self.parentsNumber,self.p)
        newPopulation = self.selectParentsAndCrossover(selected)

        self.mutation_strategy.mutate(self,newPopulation,self.mutationRate, self.lowBound,self.highBound)
        #se elitismo é verdade, pega o melhor indivíduo e passa para frente
        if self.elitism:
            best_individual = max(self.populationList, key=lambda nn: nn.fitness)
            newPopulation[0] = best_individual 

        self.populationList = newPopulation
