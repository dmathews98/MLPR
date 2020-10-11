import numpy as np
import matplotlib.pyplot as plt

def sigmoid(aa):
    return 1.0/(1.0+np.exp(-aa))

def sigmoid_gradient(aa):
    return sigmoid(aa)*(1.0-sigmoid(aa))

def central_difference(fn, ww, eps):
    return (fn(ww+eps/2.0) - fn(ww-eps/2.0))/eps

xx = np.arange(-1, 1, 0.01)
eps = 1e-5

plt.plot(xx, sigmoid_gradient(xx), 'b.')
plt.plot(xx, central_difference(sigmoid, xx, eps), 'r')
plt.show()