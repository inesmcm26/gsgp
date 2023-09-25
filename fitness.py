import numpy as np

from data import dataset, target

def sse(individual):
    print('------------------------ FITNESS -------------------------------')
    'Determine the fitness (error) of an individual. Lower is better.'

    print('INDIVIDUAL', individual.geno())

    outputs = np.apply_along_axis(lambda x: individual(*x), axis=1, arr = dataset)

    print('OUTPUTS', outputs)

    print('TARGET', target)

    sse = np.sum(np.square(outputs - target))

    print('FITNESS', sse)
    
    return sse