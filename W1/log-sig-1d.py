'''
W1C plotting Logistic-Sigmoid function in 1D
'''

import numpy as np
import matplotlib.pyplot as plt

def ls_1d(xx, vv, bb):
    return 1.0/(1 + np.exp(-xx*vv - bb))

def main():
    grid_size = 0.01
    x_vals = np.arange(-10, 10, grid_size)
    plt.clf()
    plt.plot(x_vals, ls_1d(x_vals, 0, 0), 'b.')
    plt.plot(x_vals, ls_1d(x_vals, 1, 5),'r.')
    plt.plot(x_vals, ls_1d(x_vals, 5, -5),'g.')
    plt.plot(x_vals, ls_1d(x_vals, -10, 2), 'y-')
    plt.plot(x_vals, ls_1d(x_vals, 1, 8), 'black', '.')
    plt.show()

main()

