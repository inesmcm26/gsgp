import numpy as np
import pandas as pd

from configs import NUMVARS

# dataset = np.random.randint(0, 5, size=(250, NUMVARS)).astype(float)

# target = np.sum(dataset, axis=1)


PATH = 'data/Bio/test_1.csv'

df = pd.read_csv(PATH, index_col = 0)

target = df['target'].to_numpy()

dataset = df.drop(['target'], axis = 1).to_numpy()

