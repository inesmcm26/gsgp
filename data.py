import itertools
import numpy as np

from configs import NUMVARS

dataset = np.random.randint(0, 5, size=(4, NUMVARS)).astype(float)

target = np.sum(dataset, axis=1)