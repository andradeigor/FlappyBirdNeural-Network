from abc import ABC, abstractmethod
from random import choices, sample
import numpy as np
# Interfaces
class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, populationList, parentsNumber, p):
        pass
# Seleção: Torneio
class SelectionByTournament(SelectionStrategy):
    def select(self, populationList, parentsNumber, p):
        selected = []
        for _ in range(parentsNumber):
            parent1, parent2 = sample(populationList, 2)
            if parent1.fitness > parent2.fitness:
                best, worst = parent1, parent2
            else:
                best, worst = parent2, parent1
            if np.random.rand() < p:
                selected.append(best)
            else:
                selected.append(worst)
        return selected

# Seleção: Roleta
class SelectionByRoulette(SelectionStrategy):
    def select(self, populationList, parentsNumber, p):
        populationScore = np.array([i.fitness for i in populationList])
        if populationScore.sum() == 0:
            populationProbability = np.ones(len(populationScore)) / len(populationScore)
        else:
            populationProbability = populationScore / populationScore.sum()
        selectedIndexes = np.random.choice(len(populationProbability), size=parentsNumber, p=populationProbability, replace=True)
        selected = [populationList[i] for i in selectedIndexes]
        return selected