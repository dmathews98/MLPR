import numpy as np
import matplotlib.pyplot as plt

K = int(1e6)
N = 12

xx = np.random.rand(K,N).sum(1) - 6.0

print(xx.mean())
print(xx.std())

plt.hist(xx, bins=1000)
plt.show()