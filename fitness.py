import numpy as np

from data import dataset, target

def sse(individual):
    """
    Sum of squared errors fitness function. The lowest the better.
    """

    outputs = get_outputs(individual)

    sse = np.sum(np.square(outputs - target))
    
    return sse

def mse(individual):
    """
    Mean squared error fitness function.
    """
    
    outputs = get_outputs(individual)

    return np.mean(np.square(outputs - target))

def rmse(individual):
    """
    Root mean squared error fitness function.
    """

    outputs = get_outputs(individual)
    
    return np.sqrt(np.mean(np.square(outputs - target))), outputs


def get_outputs(individual):

    outputs = np.apply_along_axis(lambda x: individual(*x), axis=1, arr = dataset)

    return outputs