from Neural import NeuralNetwork 
import numpy as np
class Genetic:
    def __init__(self,population,mutationRate,shape,parentsNumber):
        self.mutationRate = mutationRate
        self.populationList = [NeuralNetwork(shape) for x in range(population)]
        self.parentsNumber = parentsNumber
    
    def run(self,inputs):
        for NN in self.populationList:
            for input in inputs:
                prediction = NN.feedforward(input)
                if(prediction>0.5):
                    if(input[0] > 200):
                        NN.fitness+=1
                    if(input[1] < 200):
                        NN.fitness+=1
                    if(input[0]>200 and input[1] < 200):
                        NN.fitness+=5
    def selection(self):
        print("==========================================")
        populationOrderByFitness = sorted(self.populationList, key=lambda x: x.fitness, reverse=True)
        totalFitness = sum(x.fitness for x in populationOrderByFitness)
        probabilityDistribution = [populationOrderByFitness[0].fitness/totalFitness]
        for i in range(1,len(populationOrderByFitness)):
            probabilityDistribution.append(probabilityDistribution[i-1] + populationOrderByFitness[i].fitness/totalFitness)
        print("O desenpenho da população ordenado foi:")
        print("[", end="")
        for i in populationOrderByFitness:
            print(f"{i.fitness}, ", end="")
        print("]")
        print("==========================================")
        print("A probabilidade para escolher cada um é:")
        print(probabilityDistribution)
        selected = []
        for _ in range(self.parentsNumber):
            numberSelected = np.random.rand()
            for i,probability in enumerate(probabilityDistribution):
                if(numberSelected<=probability):
                    if(populationOrderByFitness[i] not in selected):
                        selected.append(populationOrderByFitness[i])
                    else:
                        selected.append(populationOrderByFitness[i+1])
                    break
        print("==========================================")
        print("Os escolhidos foram:")
        print("[", end="")
        for i in selected:
            print(f"{i.fitness}, ", end="")
        print("]")




G = Genetic(10,0.1,[2,4,4,1],2)
inputs = []
for i in range(100):
    input = [np.random.randint(100,300),np.random.randint(100,300)]
    inputs.append(input)
G.run(inputs)
G.selection()


