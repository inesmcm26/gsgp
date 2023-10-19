import random
import numpy as np

from configs import NUMVARS, DEPTH, POPSIZE

from fitness import get_outputs
from convex_hull import get_errors_matrix, is_inside_convex_hull

vars = ['x' + str(i) for i in range(NUMVARS)] # variable names

operators = ['+', '-', '*', '/']

def memoize(f):
    # receives one function f -> returns a mofidied version of the function f
    'Add a cache memory to the input function.'
    f.cache = {}

    # print('F CACHE', f.cache)

    # args are the arguments passed to f when f is invoked called
    def decorated_function(*args):
        # print('ARGS', *args)

        # If output of these arguments was already calculated, return it
        if args in f.cache:
            # print('IN CACHE')
            return f.cache[args]
        # Otherwise calculate output
        else:
            f.cache[args] = f(*args) # calculates the output of args
            # print('NOT IN CACHE')
            return f.cache[args]
                
    return decorated_function

def safe_division(x, y):
    """
    Custom function to handle division by zero
    """

    return x / y if int(y) != 0 else 1.0

def randexpr(dep):
    """
    Create a random arithmetic expression.
    """

    if dep == 1 or random.random() < 1.0 / (2**dep - 1):
        return random.choice(vars)
    
    else :
        
        # Avoid division by zero by ensuring the denominator is not zero
        op = random.choice(operators)
        if op == '/':
            left_expr = randexpr(dep-1)
            right_expr = randexpr(dep - 1)
            
            # Add a runtime check for division by zero
            return f'(safe_division({left_expr}, {right_expr}))'
        
        else:
            return '({} {} {})'.format(randexpr(dep-1),
                                     op,
                                     randexpr(dep-1))

def funct_eval(re):

    rf = eval('lambda ' + ', '.join(vars) + ': ' + re) # create function of n input variables
    rf = memoize(rf) # returns decorated function
    
    rf.geno = re # store genotype as attribute

    return rf


def randfunct():
    """
    Create a random Boolean function. Individuals are represented _directly_ as Python functions.
    """
    re = randexpr(DEPTH)
    rf = funct_eval(re)

    return rf


def check_efficiency(rf, pop):

    # print('CHECKING EFFICIENCY OF', rf.geno)

    outputs = get_outputs(rf)

    # print('OUTPUTS', outputs)

    for ind in pop:
        ind_outputs = get_outputs(ind)
        # print('POP IND OUTPUTS', ind_outputs)

        # There is one individual with the same semantics
        if np.allclose(ind_outputs, outputs, rtol=1e-5, atol=1e-8):
            return False

    return True

def check_expand(rf, pop):

    # print('CHECKING EXPAND')

    outputs = get_outputs(rf)

    errors = get_errors_matrix(pop, outputs)

    # print('ERRORS MATRIX', errors)

    is_in = is_inside_convex_hull(errors)

    # print('INSIDE CONVEX HULL', is_in)

    if is_in:
        return False
    else:
        return True


def competent_initialization():
    # The algorithm begins with filling the working initial population P with single-node
    # programs made of individual terminal instructions

    pop = []

    for var in vars:
        re = f'({var})'
        rf = funct_eval(re)
        pop.append(rf)

    # print('INITIAL POP WITH TERMINALS')
    # for ind in pop:
    #     print(ind.geno)

    failed_attemps = 0

    # Builds new program candidates bottom-up from the programs already present in P 
    while len(pop) < POPSIZE:
        
        # Choose two random programs from P
        left_expr = random.choice(pop) # rf
        # print('LEFT EXPR', left_expr.geno)
        right_expr = random.choice(pop) # rf
        # print('RIGHT EXPR', right_expr.geno)

        op = random.choice(operators)
        # print('OP:', op)

        if op == '/':
            rf = lambda *x: safe_division(left_expr(*x), right_expr(*x))
            rf = memoize(rf)
            rf.geno = f'(safe_division({left_expr.geno}, {right_expr.geno}))'
        elif op == '+':
            rf = lambda *x: left_expr(*x) + right_expr(*x)
            rf = memoize(rf)
            rf.geno = f'({left_expr.geno} + {right_expr.geno})'
        elif op == '-':
            rf = lambda *x: left_expr(*x) - right_expr(*x)
            rf = memoize(rf)
            rf.geno = f'({left_expr.geno} - {right_expr.geno})'
        elif op == '*':
            rf = lambda *x: left_expr(*x) * right_expr(*x)
            rf = memoize(rf)
            rf.geno = f'({left_expr.geno} * {right_expr.geno})'

        # print('NEW IND', rf.geno)

        # rf = eval(f'lambda *x: left_expr(*x) {op} right_exp(*x)')

        # rf = funct_eval(re)

        # Check if rf is semantically different from all programs in P
        # Check if rf expands convex hull
        if check_efficiency(rf, pop) and check_expand(rf, pop):
            pop.append(rf)
            failed_attemps = 0
            # print('NEW POP')
            # for ind in pop:
            #     print(ind.geno)
        elif failed_attemps >= 50:
            print('50 failed attemps')
            pop.append(rf)
            failed_attemps = 0
        else:
            failed_attemps = failed_attemps + 1

        # print('--------------------------------')

    return pop
