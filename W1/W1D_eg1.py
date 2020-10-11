import numpy as np
import matplotlib.pyplot as plt

plt.clf()

def phi_linear(Xin):
    return np.hstack([np.ones((Xin.shape[0],1)), Xin])
def phi_quadratic(Xin):
    return np.hstack([np.ones((Xin.shape[0],1)), Xin, Xin**2])
def fw_rbf(xx, cc):
    """fixed-width RBF in 1d"""
    return np.exp(-(xx-cc)**2 / 0.25)
def phi_rbf(Xin):
    return np.hstack([fw_rbf(Xin, 0.25), fw_rbf(Xin, 0.5), fw_rbf(Xin, 0.75)])

def fit_and_plot(phi_fn, X, yy):
    # phi_fn takes N, inputs and returns N,D basis function values
    w_fit = np.linalg.lstsq(phi_fn(X), yy, rcond=None)[0] # D of X plus 1 for linear,
    print(w_fit)
    print(np.mean(w_fit))
    X_grid = np.tile(np.arange(0, 1, 0.01)[:,None], (1,D))
    f_grid = np.dot(phi_fn(X_grid), w_fit)
    plt.plot(np.arange(0, 1, 0.01)[:,None], f_grid, linewidth=2)

N = 505
D = 500

mu = np.random.rand(N)
X = np.tile(mu[:,None], (1, D)) + 0.01*np.random.randn(N, D)
yy = 0.1*np.random.randn(N) + mu

plt.plot(X[:,0], yy, 'x', markersize=20, mew=2)
fit_and_plot(phi_linear, X, yy)
# fit_and_plot(phi_quadratic, X, yy)
# fit_and_plot(phi_rbf, X, yy)
plt.legend(('data', 'linear fit', 'quadratic fit', 'rbf fit'))
plt.xlabel('x')
plt.ylabel('f')

# plt.show()