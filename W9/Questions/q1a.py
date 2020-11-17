import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

N = 10
data_0 = np.hstack([1.5+0.75*np.random.randn(N,1), 1.5+0.25*np.random.randn(N,1)])
data_1 = np.hstack([1.5+0.75*np.random.randn(N,1), 0.5+0.25*np.random.randn(N,1)])

test = np.hstack([1.5+np.random.randn(5,1), 1.0+np.random.randn(5,1)])

plt.figure(1)
plt.scatter(data_0[:,0], data_0[:,1], color='red')
plt.scatter(data_1[:,0], data_1[:,1], color='blue')
plt.scatter(test[:,0], test[:,1], color='black')

A = np.array([[0, 0],
              [0, 1]])

data_0 = np.transpose(A @ (data_0.T))
data_1 = np.transpose(A @ (data_1.T))
test = np.transpose(A @ (test.T))

plt.figure(2)
plt.scatter(data_0[:,0], data_0[:,1], color='red')
plt.scatter(data_1[:,0], data_1[:,1], color='blue')
plt.scatter(test[:,0], test[:,1], color='black')
plt.show()