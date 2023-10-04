import cvxpy as cp
import numpy as np

from data import dataset, target

def is_inside_convex_hull(ind_list):

    errors = get_errors_matrix(ind_list)

    M, N = errors.shape  # Get the shape of the matrix x

    a = cp.Variable(N, nonneg=True)  # Define a as a non-negative variable

    # Formulate the optimization problem
    constraints = [
        cp.sum(a) == 1,
        errors @ a == np.zeros(M)  # Corrected equation
    ]

    objective = cp.Minimize(0)  # Feasibility problem

    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve()

    if prob.status == cp.OPTIMAL:
        print("Feasible solution found:", a.value)
        return True
    else:
        print("No feasible solution found.")
        return False


def get_errors_matrix(ind_list):
    M = len(target)
    N = len(ind_list)

    errors = np.zeros((M, N))

    for idx, individual in enumerate(ind_list):
        outputs = np.apply_along_axis(lambda x: individual(*x), axis=1, arr = dataset)

        # print(f'OUTPUTS OF IND {individual.geno()}')
        # print(outputs)

        errors[:, idx] = target - outputs

    return errors


def create_problem_go_in_ch(errors):
    """
    errors (np.array): matrix of the errors. Shape M x N -> dimensions x individuals
    """

    M, N = errors.shape

    X = errors
    ZEROS = np.zeros(M)

    X_PAR = cp.Parameter(X.shape)
    ZEROS_PAR = cp.Parameter(ZEROS.shape)

    a = cp.Variable(N)

    objective = cp.Minimize(cp.sum_squares(X_PAR@a - ZEROS_PAR))

    constraints = [cp.sum(a) == 1]

    # Create additional constraints to ensure ai >= 0 for all i
    for i in range(N):
        constraints.append(a[i] >= 0)

    prob = cp.Problem(objective, constraints)

    if prob.status == cp.OPTIMAL:
        print("Feasible solution found.")
        return True
    elif prob.status == cp.INFEASIBLE:
        print("The problem is infeasible. No solution exists.")
        return False
    elif prob.status == cp.UNBOUNDED:
        print("The problem is unbounded. There are infinitely many solutions.")
        return False
    else:
        raise Exception("The solver encountered an issue.")

 

def solve_problem_go_in_ch(prob, errors):

    A = errors.T.numpy()

    b = np.zeros(A.shape[0])

    prob.parameters()[0].value = A

    prob.parameters()[1].value = b

    try:

        prob.solve( solver = 'MOSEK')

        # print(f'Sum equal to 1 {np.isclose(sum(prob.variables()[0].value),1)}')

        # print(f'All positive: {all(prob.variables()[0].value >= 0)}')

        return prob.status == 'optimal' and np.isclose(sum(prob.variables()[0].value),1) and all(prob.variables()[0].value >= 0)

    except:

        print('Optimization not completed')

        return False