'''
W1C plotting RBF
'''
import numpy as np
import matplotlib.pyplot as plt

def func_1(xx):
    return 2.0*rbf_1d(xx, -5, 1) - rbf_1d(xx, 5, 1)

def rbf_1d(xx, cc, hh):
    return np.exp(-(xx-cc)**2 / hh**2)

plt.clf()
grid_size = 0.01
x_grid = np.arange(-10, 10, grid_size)
plt.plot(x_grid, func_1(x_grid))
plt.show()