import cvxpy as cp
import numpy as np

from data import dataset

def is_inside_convex_hull(errors):

    M, N = errors.shape  # Get the shape of the matrix x

    a = cp.Variable(N, nonneg=True)  # Define a as a non-negative variable

    # Formulate the optimization problem
    constraints = [
        cp.sum(a) == 1,
        errors @ a == np.zeros(M)  # Corrected equation
    ]

    objective = cp.Minimize(0)  # Feasibility problem

    prob = cp.Problem(objective, constraints)

    try:
        # Solve the problem
        prob.solve()

        if prob.status == cp.OPTIMAL:
            # print("Feasible solution found:", a.value)
            return True
        else:
            # print("No feasible solution found.")
            return False
    except cp.error.SolverError:
        return False


def get_errors_matrix(ind_list, target):
    M = len(target)
    N = len(ind_list)

    errors = np.zeros((M, N))

    for idx, individual in enumerate(ind_list):
        outputs = np.apply_along_axis(lambda x: individual(*x), axis=1, arr = dataset)

        # print(f'OUTPUTS OF IND {individual.geno}')
        # print(outputs)

        errors[:, idx] = target - outputs

    return errors