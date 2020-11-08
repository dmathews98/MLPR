import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sc

def log_sig(a):
    return 1.0/(1.0+np.exp(-a))

N = 100
D_w = 5
H = 3

xx = np.random.rand(N)
w1 = np.random.uniform(low=-1.0, high=1.0, size=(D_w))
W2 = np.random.uniform(low=-1.0, high=1.0, size=(D_w,D_w))
W3 = np.random.uniform(low=-1.0, high=1.0, size=(D_w,D_w))
w4 = np.random.uniform(low=-1.0, high=1.0, size=(D_w))
ff = np.zeros(N)

for i in range(N):
    h1 = log_sig(xx[i]*w1.T)
    h2 = log_sig(W2@h1)
    h3 = log_sig(W3@h2)
    ff[i] = log_sig(w4@h3)

plt.plot(xx, ff, '.')
plt.show()
