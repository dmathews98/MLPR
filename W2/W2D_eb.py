import numpy as np
import matplotlib.pyplot as plt

N = 10000

# xx = 1 * (np.random.rand(N) < 0.3)
xx = np.random.randn(N)
print(xx)

mu = np.mean(xx)
sigma = np.std(xx)

sem = sigma/np.sqrt(N)

print(mu)
print(sigma)
print(sem)

plt.clf()
plt.hist(xx, 100)
plt.vlines([mu, mu+sigma, mu-sigma, mu+sem, mu-sem], 0, 300, ['black', 'red', 'red', 'green', 'green'])
plt.show()