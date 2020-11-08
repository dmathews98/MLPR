import numpy as np
import matplotlib.pyplot as plt

Sigma_inv = np.array([[1.0, 0.0], [0.0, 1.0]])
mu = np.array([[0.5], [0.5]])

A = Sigma_inv/2.0
cc = -2.0 * (A @ mu)

def func_1(x, y):
    vec = np.array([[x], [y]])
    return np.exp(-(vec.T @ (A @ vec)) - (vec.T @ cc)) 

def func_2(x, y):
    vec = np.array([[x], [y]])
    return np.exp(-0.5 * (np.transpose(vec - mu) @ (Sigma_inv @ (vec - mu))) ) 

x = np.arange(-3, 3, 0.1)
y = np.arange(-3, 3, 0.1)

unnorm_1 = np.zeros((len(x), len(y)))
unnorm_2 = np.zeros((len(x), len(y)))

for i in range(len(x)):
    for j in range(len(y)):
        unnorm_1[i,j] = func_1(x[i], y[j])
        unnorm_2[i,j] = func_2(x[i], y[j])

res_1 = unnorm_1/np.sum(unnorm_1)
res_2 = unnorm_2/np.sum(unnorm_2)

sum_sq_diff = np.sum((res_1 - res_2)**2.0)
print(sum_sq_diff)

plt.figure(1)
plt.contourf(x,y,res_1)
plt.axis('square')
plt.figure(2)
plt.contourf(x, y, res_2)
plt.axis('square')
plt.show()