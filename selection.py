import random

from fitness import rmse as fitness

def tournament(individuals, k):
    """
    Tournament selection.

    Args:
        - individuals (list): list of individuals (higher order functions)
        - k (int): tournament size
    """

    # Randomly select k individuals with replacement
    tournament = random.choices(individuals, k=k)

    # print('TOURNAMENT')
    # for ind in tournament:
    #     print('IND', ind.geno(), 'FITNESS', fitness(ind)[0])

    # Return the best individual
    best_ind = min(tournament, key = lambda x: fitness(x)[0])

    # print('BEST IND', best_ind.geno(), 'FITNESS', fitness(best_ind)[0])

    return best_ind