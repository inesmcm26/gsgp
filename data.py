import itertools
import numpy as np

from configs import NUMVARS

# somelists = [[True,False] for i in range(NUMVARS)]

# dataset = np.array(list(itertools.product(*somelists)))

# target = [True if i%2 == 0 else False for i in range(8)]

dataset = np.random.randint(0, 5, size=(4, NUMVARS))

target = np.sum(dataset, axis=1)
