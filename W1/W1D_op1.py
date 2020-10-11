import numpy as np
import matplotlib.pyplot as plt

def fw_rbf(xx, cc):
    """fixed-width RBF in 1d"""
    return np.exp(-(xx-cc)**2 / 0.2)
def phi_rbf(Xin):
    ccs = np.linspace(-0.5, 0.5, 12)
    res = fw_rbf(Xin, ccs[0])
    for i in range(1, len(ccs)):
        res = np.hstack([res, fw_rbf(Xin, ccs[i])])
    return res

def fit_and_plot(phi_fn, X, yy):
    # phi_fn takes N, inputs and returns N,D basis function values
    w_fit = np.linalg.lstsq(phi_fn(X), yy, rcond=None)[0] # D,
    X_grid = np.arange(-0.5, 0.5, 0.01)[:,None] # N,1
    f_grid = np.dot(phi_fn(X_grid), w_fit)
    plt.plot(X_grid, f_grid, linewidth=1)

def regularized_fit(phi_fn, X, yy, hh):
    y_tilde = np.vstack([yy[:,None], np.zeros((12,1))])   # (N+K),1
    phi_tilde = np.vstack([phi_fn(X), np.sqrt(hh)*np.identity(12)])   # (N+K), K
    w_fit = np.linalg.lstsq(phi_tilde, y_tilde, rcond=None)[0]   #  K,1
    X_grid = np.arange(-0.5, 0.5, 0.01)[:,None] 
    f_grid = np.dot(phi_fn(X_grid), w_fit) 
    plt.plot(X_grid, f_grid, linewidth=1)


yy = np.array([-1.0, -1.0, 0, 0.4, 0.1, -0.1, -0.3, 0.5, 0.4, 0.2, -0.3, 0.5])  # N,
X = np.array([-0.45, -0.3, -0.1, -0.09, -0.06, 0.1, 0.11, 0.3, 0.35, 0.4, 0.42, 0.47])[:,None]

plt.clf()
plt.plot(X[:,0], yy, 'x', markersize=5, mew=1)

fit_and_plot(phi_rbf, X, yy)

hh = 0.5   # lambda
regularized_fit(phi_rbf, X, yy, hh)

plt.ylim(-4.5, 2)
plt.legend(('data', 'lstsq fit', 'regularized fit'))
plt.xlabel('x')
plt.ylabel('y')
plt.show()