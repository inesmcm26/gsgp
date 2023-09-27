import random
from math import exp, sqrt
import numpy as np

from configs import MUTATION_STEP
from data import target

from generators import memoize, randfunct

from fitness import rmse as fitness

from data import dataset

def crossover(p1,p2):
    """
    The crossover operator is a higher order function that takes parent functions and return an offspring function.
    The definitions of parent functions are _not substituted_ in the definition of the offspring function.
    Instead parent functions are _called_ from the offspring function. This prevents exponential growth. 

    Args:
        p1: higher order function
        p2: higher order function
    Returns:
        offspring: higher order function -> C = (P1·RF) + (1-RF)·P2
    """

    # print('----------------------- CROSSOVER ----------------------------')

    # print('PARENTS')
    # print('P1', p1.geno())
    # print('P2', p2.geno())
    
    random_num = random.uniform(0, 1)

    offspring = lambda *x: ((p1(*x) * random_num) + (p2(*x) * (1.0 - random_num)))  # Define arithmetic crossover operation
    offspring = memoize(offspring) # add cache

    offspring.geno = lambda: '(({} * sigmoid({})) + ({} * (1 - sigmoid({}))))'.format(p1.geno(), random_num, p2.geno(), random_num)
    
    # print('OFFSPRING', offspring.geno(), 'FITNESS', fitness(offspring)[0])

    # if fitness(offspring)[0] > fitness(p1)[0] and fitness(offspring)[0] > fitness(p2)[0]:

    #     print('RANDOM NUMBER', random_num)

    #     print('1 - RANDOM NUMBER', 1.0 - random_num)

    #     # print('EQUAL PARENTS', p1.geno() == p2.geno())

    #     print('EQUAL PARENTS OUTPUT', np.all(fitness(p1)[1] == fitness(p2)[1]))
    #     print('EQUAL CHILD AND P1 OUTPUTS', np.all(fitness(p1)[1] == fitness(offspring)[1]))

    #     print('P1 OUTPUT HARDCODED', p1(*dataset[0]), ',', p1(*dataset[1]))
    #     print('P2 OUTPUT HARDCODED', p2(*dataset[0]), ',', p2(*dataset[1]))

    #     print('FIRST OBS CHILD', (p1(*dataset[0]) * random_num) + (p2(*dataset[0]) * (1.0 - random_num)))
    #     print('CHILD OUTPUT HARDCODED: [', (p1(*dataset[0]) * random_num) + (p2(*dataset[0]) * (1.0 - random_num)), ',', \
    #           (p1(*dataset[1]) * random_num) + (p2(*dataset[1]) * (1 - random_num)))

    #     print('P1 FITNESS', fitness(p1)[0])
    #     print('P2 FITNESS', fitness(p2)[0])
    #     print('OFFSPRING FITNESS', fitness(offspring)[0])

    #     print('TARGET', target)
    #     print('OUTPUT CHILDREN', fitness(offspring)[1])
    #     print('OUTPUT P1', fitness(p1)[1])
    #     print('OUTPUT P2', fitness(p2)[1])

    #     raise Exception('Crossover anomaly!!!')
    
    return offspring

def sigmoid(x):
    try:
        return exp(x) / (1.0 + exp(x))
    except OverflowError:
        return 1.0


def mutation(p):
    """
    The mutation operator is a higher order function. The parent function is called by the offspring function.

    Args:
        p: higher order function
    Returns:
        c: higher order function -> C = P + (RF1 - RF2) * ms
    """
    # print('----------------------- MUTATION ----------------------------')

    # print('PARENT', p.geno())

    rf1 = randfunct()
    rf2 = randfunct()

    # print('RF1', rf1.geno())
    # print('RF2', rf2.geno())

    ms = random.uniform(0, MUTATION_STEP)

    # print('MS:', ms)

    offspring = lambda *x: (p(*x) + (ms * sigmoid((rf1(*x) - rf2(*x)))))  # Define arithmetic crossover operation
    offspring = memoize(offspring)
    offspring.geno = lambda: '({} + ({} * sigmoid(({} - {}))))'.format(p.geno(), ms, rf1.geno(), rf2.geno())

    # print('MUTATION OFFSPRING', offspring.geno(), 'FITNESS', fitness(offspring)[0])
    return offspring