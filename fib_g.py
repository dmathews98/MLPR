import numpy as np
import matplotlib.pyplot as plt

def get_next_fib(prev2, prev1):
    return prev2+prev1

fib = [1.0, 1.0]
ratio = [1.0]
count = 100

for i in range(count):
    next_fib = get_next_fib(fib[i], fib[i+1])
    fib.append(next_fib)
    ratio.append(next_fib/fib[i+1])

plt.plot(ratio)
plt.show()