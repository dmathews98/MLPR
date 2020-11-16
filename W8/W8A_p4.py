import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

def sigmoid(a): return 1. / (1. + np.exp(-a))
def relu(x): return np.maximum(x, 0)
def linear(a): return a

def neural_net(X, layer_sizes=(100,50,50,1), gg=relu, sigma_w=1, sigma_b=1):
    for out_size in layer_sizes:
        Wt = sigma_w * np.random.randn(X.shape[1], out_size)
        bb = sigma_b * np.random.randn(1, out_size)
        X = gg(X @ Wt + bb)
        # X = gg(X @ Wt) 
    return X

N = 100
X = np.linspace(-2, 2, num=N)[:, None]  # N,1
plt.clf()
for i in range(5):
    ff = neural_net(X)
    plt.plot(X, ff)
    plt.show()

