import numpy as np
import matplotlib.pyplot as plt

def phi_k(xx, ck):
    return (xx-ck)**2.0

def rbf_1d(xx, cc, hh):
    return np.exp(-(xx-cc)**2 / hh**2)

kk = 10
c = np.arange(-10, 10, 2)
ww = np.random.randn(kk)
x_grid = np.arange(-10, 10, 0.1)

phi_mat_q = np.empty((kk, len(x_grid)))
phi_mat_rbf = np.empty((kk, len(x_grid)))

phi_mat = np.empty((kk, len(x_grid)))

for i in range(kk):
    phi_mat_q[i] = phi_k(x_grid, c[i])
    phi_mat_rbf[i] = rbf_1d(x_grid, c[i], 1)

    if np.random.random()<0.5:
        phi_mat[i] = phi_k(x_grid, c[i])
    else:
        phi_mat[i] = rbf_1d(x_grid, c[i], 1)

plt.clf()
plt.plot(x_grid, np.dot(ww,phi_mat_q),'b')
plt.plot(x_grid, np.dot(ww,phi_mat_rbf),'r')
plt.plot(x_grid, np.dot(ww,phi_mat),'g')
plt.show()