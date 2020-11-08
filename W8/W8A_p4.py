import numpy as np
from matplotlib import pyplot as plt

# def log_sig(a):
#     return 1.0/(1.0+np.exp(-a))

# def neural_net(X):
#     H1 = 100
#     H2 = 50
#     W1 = np.random.uniform(low=-1.0, high=1.0, size=(H1, len(X)))
#     W2 = np.random.uniform(low=-1.0, high=1.0, size=(H2, H1))
#     W3 = np.random.uniform(low=-1.0, high=1.0, size=(len(X), H2))

#     h1 = log_sig(W1@X)
#     h2 = log_sig(W2@h1)
#     ff = log_sig(W3@h2)

#     return ff

def sigmoid(a): return 1. / (1. + np.exp(-a))
def relu(x): return np.maximum(x, 0)
def linear(a): return a

def neural_net(X, layer_sizes=(100,50,1), gg=sigmoid, sigma_w=1):
    for out_size in layer_sizes:
        Wt = sigma_w * np.random.randn(X.shape[1], out_size)
        X = gg(X @ Wt)
    return X

N = 100
X = np.linspace(-2, 2, num=N)[:, None]  # N,1
plt.clf()
for i in range(12):
    ff = neural_net(X)
    plt.plot(X, ff)
    plt.show()

