
'''

TINY_GSGP.py: A Tiny and Efficient Implementation of Geometric Semantic Genetic Programming Using Higher-Order Functions and Memoization

Author: Alberto Moraglio (albmor@gmail.com) 

Features:

- Individuals are represented directly as Python (anonymous) functions.

- Crossover and mutation are higher-order functions.

- Offspring functions call parent functions rather than embed their definitions (no grwoth, implicit ancestry trace).

- Memoization of individuals turns time complexity of fitness evalutation from exponential to constant.

- The final solution is a compiled function. It can be extracted using the ancestry trace to reconstruct its 'source code'. 

This implementation is to evolve Boolean expressions. It can be easily adapted to evolve arithmetic expressions or classifiers.

'''

import random
import numpy as np

from configs import NUMVARS, GENERATIONS, POPSIZE, TRUNC

from data import dataset, target
from generators import randfunct
from operators import mutation, crossover

vars = ['x'+str(i) for i in range(NUMVARS)] # variable names

print(dataset)
print(target)


def fitness(individual):
    print('------------------------ FITNESS -------------------------------')
    'Determine the fitness (error) of an individual. Lower is better.'

    print('INDIVIDUAL', individual.geno())

    outputs = np.apply_along_axis(lambda x: individual(*x), axis=1, arr = dataset)

    print('OUTPUTS', outputs)

    print('TARGET', target)

    sse = np.sum(np.square(outputs - target))

    print('FITNESS', sse)
    
    return sse
    
def evolve():
    'Main function.'
    pop = [randfunct() for _ in range(POPSIZE)] # initialise population

    print('------------------------ POPULATION INITIALIZED ------------------------')

    for gen in range(GENERATIONS+1):
        print()
        print(f'------------------------------------------ GENERATION {gen} ----------------------------------------------')
        
        graded_pop = [(fitness(ind), ind) for ind in pop] # evaluate population fitness

        for ind in graded_pop:
            print('IND', ind[1].geno(), 'FITNESS', ind[0])

        sorted_pop = [ind[1] for ind in sorted(graded_pop, key = lambda x: x[0])] # sort population on fitness

        print('gen: ', gen , ' min fit: ', fitness(sorted_pop[0]), ' avg fit: ', sum(ind[0] for ind in graded_pop)/(POPSIZE)) # print stats
        
        parent_pop = sorted_pop[:int(TRUNC*POPSIZE)] # selected parents
        
        if gen == GENERATIONS:
            break
        
        for i in range(POPSIZE): # create offspring population
            par = random.sample(parent_pop, 2) # pick two random parents

            pop[i] = crossover(par[0],par[1]) # create offspring

    best_ind = sorted_pop[0]
    best_fitness = fitness(sorted_pop[0])

    print('Best individual in last population:', best_ind.geno(), ' \n With fitness:', best_fitness)
    print('Predictions\n', np.apply_along_axis(lambda x: best_ind(*x), axis=1, arr = dataset))

evolve()

