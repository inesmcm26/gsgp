import itertools
import numpy as np

NUMVARS = 3 # number of input variables

somelists = [[True,False] for i in range(NUMVARS)]

dataset = np.array(list(itertools.product(*somelists)))

target = [True if i%2 == 0 else False for i in range(8)]
