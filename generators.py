import random

from configs import NUMVARS, DEPTH

vars = ['x'+str(i) for i in range(NUMVARS)] # variable names

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

def randfunct():
    """
    Create a random Boolean function. Individuals are represented _directly_ as Python functions.
    """
    re = randexpr(DEPTH)

    rf = eval('lambda ' + ', '.join(vars) + ': ' + re) # create function of n input variables
    rf = memoize(rf) # returns decorated function
    
    rf.geno = lambda: re # store genotype as attribute

    return rf