import numpy as np
import matplotlib.pyplot as plt

def plot_single_run(history):
    runs = np.array([i for i in range(len(history))])

    plt.figure(figsize=(20, 10))

    plt.plot(runs, history, linestyle='-')

    # Add labels to the axes
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    # Add a title to the plot
    plt.title('Fitness Over Generations')

    plt.show()