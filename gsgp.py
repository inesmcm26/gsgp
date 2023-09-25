
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
import itertools
import numpy as np

from data import dataset, target

#### PARAMETERS ####

NUMVARS = 3 # number of input variables
DEPTH = 4 # maximum depth of expressions in the initial population
POPSIZE = 5 # population size
GENERATIONS = 2 # number of generations
TRUNC = 0.5 # proportion of population to retain in truncation selection

####################

vars = ['x'+str(i) for i in range(NUMVARS)] # variable names

print(dataset)
print(target)

def memoize(f):
    # receives one function f -> returns a mofidied version of the function f
    'Add a cache memory to the input function.'
    f.cache = {}

    # print('F CACHE', f.cache)

    # args are the arguments passed to f when f is invoked called
    def decorated_function(*args):

        print('CACHE', f.cache)
        print('ARGS', args)
        print('ARGS POINTER', *args)

        # If output of these arguments was already calculated, return it
        if args in f.cache:
            print('IN CACHE', f.cache[args])
            return f.cache[args]
        # Otherwise calculate output
        else:
            f.cache[args] = f(*args) # calculates the output of args
            print('NOT IN CACHE. ADDED:', f.cache[args])
            return f.cache[args]
        
    return decorated_function

def randexpr(dep):
    'Create a random Boolean expression.'
    # When reached final depth or probability is met
    if dep==1 or random.random()<1.0/(2**dep-1):
        return random.choice(vars)
    
    # 30% chance of happening
    if random.random() < 1.0/3:
        # Negated recursive call
        return 'not' + ' ' + randexpr(dep-1) 
    
    # Recursive call
    else:
        return '(' + randexpr(dep-1) + ' ' + random.choice(['and','or']) + ' ' + randexpr(dep-1) + ')'

def randfunct():
    'Create a random Boolean function. Individuals are represented _directly_ as Python functions.'
    re = randexpr(DEPTH)
    # print('RANDOM EXPRESSION:', re)

    # print('TO EVAL', 'lambda ' + ', '.join(vars) + ': ' + re): lambda x0, x1, x2, x3, x4: not not x0
    rf = eval('lambda ' + ', '.join(vars) + ': ' + re) # create function of n input variables
    # print('EVAL RF', rf)
    rf = memoize(rf) # returns decorated function
    rf.geno = lambda: re # store genotype as attribute

    # print('RANDOM FUNCTION', rf)
    # print('RF GENOTYPE', rf.geno())

    return rf

# def randarithexpr(dep):
#     'Create a random arithmetic expression.'
#     if dep == 1 or random.random() < 1.0 / (2 ** dep - 1):
#         return random.choice(vars)
#     op = random.choice(['+', '-', '*', '/'])
#     return '(' + randarithexpr(dep - 1) + ' ' + op + ' ' + randarithexpr(dep - 1) + ')'

# def randarithfunct():
#     'Create a random arithmetic function. Individuals are represented _directly_ as Python functions.'
#     re = randarithexpr(DEPTH)
#     rf = eval('lambda ' + ', '.join(vars) + ': ' + re)  # create function of n input variables
#     rf = memoize(rf)  # add cache to the function
#     rf.geno = lambda: re  # store genotype
#     return rf


def targetfunct(*args):
    'Parity function of any number of input variables'
    return args.count(True) % 2 == 1

def fitness(individual):
    'Determine the fitness (error) of an individual. Lower is better.'
    fit = 0
    somelists = [[True,False] for i in range(NUMVARS)] # -> This is the dataset!

    print('INDIVIDUAL', individual.geno())

    outputs = np.apply_along_axis(lambda x: individual(*x), axis=1, arr = dataset)

    print('OUTPUTS', outputs)

    print('TARGET', target)

    fit = np.sum(outputs != target)

    print('FITNESS', fit)
    
    return fit

def crossover(p1,p2):
    """
    The crossover operator is a higher order function that takes parent functions and return an offspring function.
    The definitions of parent functions are _not substituted_ in the definition of the offspring function.
    Instead parent functions are _called_ from the offspring function. This prevents exponential growth.    
    """
    mask = randfunct()
    offspring = lambda *x: (p1(*x) and mask(*x)) or (p2(*x) and not mask(*x))
    offspring = memoize(offspring) # add cache
    offspring.geno = lambda: '(('+ p1.geno() + ' and ' + mask.geno() + ') or (' + p2.geno() + ' and not ' + mask.geno() + '))' # to reconstruct genotype
    return offspring

def mutation(p):
    'The mutation operator is a higher order function. The parent function is called by the offspring function.'
    mintermexpr = ' and '.join([random.choice([x,'not ' + x]) for x in vars]) # random minterm expression of n variables
    minterm = eval('lambda ' + ', '.join(vars) + ': ' + mintermexpr) # turn minterm into a function
    if random.random()<0.5:
        offspring = lambda *x: p(*x) or minterm(*x)
        offspring = memoize(offspring) # add cache
        offspring.geno = lambda: '(' + p.geno() + ' or ' + mintermexpr + ')' # to reconstruct genotype
    else:
        offspring = lambda *x: p(*x) and not minterm(*x)
        offspring = memoize(offspring) # add cache
        offspring.geno = lambda: '(' + p.geno() + ' and not ' + mintermexpr + ')' # to reconstruct genotype
    return offspring
    
def evolve():
    'Main function.'
    pop = [randfunct() for _ in range(POPSIZE)] # initialise population

    for gen in range(GENERATIONS+1):
        graded_pop = [(fitness(ind), ind) for ind in pop] # evaluate population fitness

        for ind in graded_pop:
            print('IND', ind[1].geno(), 'FITNESS', ind[0])

        sorted_pop = [ind[1] for ind in sorted(graded_pop, key = lambda x: x[0])] # sort population on fitness
        break
        print('gen: ', gen , ' min fit: ', fitness(sorted_pop[0]), ' avg fit: ', sum(ind[0] for ind in graded_pop)/(POPSIZE)) # print stats
        parent_pop = sorted_pop[:int(TRUNC*POPSIZE)] # selected parents
        if gen == GENERATIONS: break
        for i in range(POPSIZE): # create offspring population
            par = random.sample(parent_pop, 2) # pick two random parents
            pop[i] = mutation(crossover(par[0],par[1])) # create offspring

    best_ind = sorted_pop[0]

    print('Best individual in last population: ', best_ind.geno())
    #print (sorted_pop[0]).geno() # reconstruct genotype of final solution (WARNING: EXPONENTIALLY LONG IN NUMBER OF GENERATIONS!)
    print('Query best individual in last population with all True inputs:')    
    print('Predictions\n', np.apply_along_axis(lambda x: best_ind(*x), axis=1, arr = dataset))

evolve()

