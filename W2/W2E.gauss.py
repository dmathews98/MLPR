import numpy as np
import matplotlib.pyplot as plt

N = int(1e4); D = 2
X = np.random.randn(N, D)

aa = 0.4
bb = 0.8
# A = np.array([[1, 0],[aa, 1-aa]])

A = np.array([[aa, 0.2], [0.3, bb]])

Sigma = np.dot(A, A.T)

Y = np.transpose(np.dot(A, X.T))

sigma_check = np.cov(Y.T)

print(Sigma)
print(sigma_check)

# print(np.linalg.det(np.cov(X.T)))
# D = 3; Sigma = np.cov(np.random.randn(D, 3*D))
# A = np.linalg.cholesky(Sigma)
# Sigma_from_A = np.dot(A, A.T)  # up to round-off error, matches Sigma
# print(A)
# print(Sigma_from_A)

plt.plot(X[:,0], X[:,1], 'b.')
plt.plot(Y[:,0], Y[:,1], 'r.')
plt.axis('square')
plt.show()