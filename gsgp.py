import random
import numpy as np

from configs import NUMVARS, GENERATIONS, POPSIZE, MUT_PROB, XO_PROB, TOURNAMENT_SIZE

from data import dataset, target
from generators import randfunct, competent_initialization
from operators import mutation, crossover
from fitness import rmse as fitness
from selection import tournament
from convex_hull import is_inside_convex_hull, get_errors_matrix

from plots import plot_single_run

vars = ['x'+str(i) for i in range(NUMVARS)] # variable names

print(dataset)
print(target)

fitness_history = []
    
def evolve():
    'Main function.'
    # pop = [randfunct() for _ in range(POPSIZE)] # initialize population

    inside_count = 0

    for _ in range(1000):
        pop = competent_initialization()

        errors = get_errors_matrix(pop, target)
        is_inside = is_inside_convex_hull(errors)

        if is_inside:
            inside_count = inside_count + 1
        # print(is_inside)

    print(inside_count)

    return

    for gen in range(GENERATIONS+1):
        print()
        print(f'------------------- GENERATION {gen} -----------------------')
        
        graded_pop = [(fitness(ind)[0], ind) for ind in pop] # Evaluate population fitness
        
        # print('---------------------------')
        # for ind in graded_pop:
        #     print('IND', ind[1].geno, 'FITNESS', ind[0])
        # print('---------------------------')

        sorted_pop = [ind[1] for ind in sorted(graded_pop, key = lambda x: x[0])] # Sort population on fitness
        
        best_fitness = fitness(sorted_pop[0])[0]
        fitness_history.append(best_fitness)

        print('Min fit: ', best_fitness, ' Avg fit: ', sum(ind[0] for ind in graded_pop)/(POPSIZE)) # print stats
        
        if gen == GENERATIONS:
            break

        if fitness(sorted_pop[0])[0] == 0.0:
            break
        
        # Create new population
        new_pop = []

        for _ in range(POPSIZE):
            # Genetic operators
            random_number = random.random()

            # Mutation
            if random_number < MUT_PROB:
                # print('MUTATION')
                p = tournament(pop, TOURNAMENT_SIZE)
                child = mutation(p)
            # Crossover
            elif random_number < MUT_PROB + XO_PROB:
                # print('CROSSOVER')
                p1 = tournament(pop, TOURNAMENT_SIZE)
                p2 = tournament(pop, TOURNAMENT_SIZE)
                child = crossover(p1, p2)
            # Replication
            else:
                # print('REPLICATION')
                child = tournament(pop, TOURNAMENT_SIZE)
                # print('CHILD', child.geno)

            # Add child to population
            new_pop.append(child)
        
        pop = new_pop

        # print('NEW POPULATION')
        # for ind in pop:
        #     print('IND', ind.geno)

    best_ind = sorted_pop[0]
    best_fitness = fitness(sorted_pop[0])[0]

    print('Best individual in last population:', #best_ind.geno,
          ' \n With fitness:', best_fitness)
    print('Predictions\n', np.apply_along_axis(lambda x: best_ind(*x), axis=1, arr = dataset))

evolve()


# plot_single_run(fitness_history)