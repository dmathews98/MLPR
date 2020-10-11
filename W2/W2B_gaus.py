import numpy as np
import matplotlib.pyplot as plt

N  = int(1e6)
xx = np.random.randn(N)

plt.clf()
hist_stuff = plt.hist(xx,100)

print('empirical_mean = %g' % np.mean(xx)) # or xx.mean()
print('empirical_var = %g' % np.var(xx))   # or xx.var()

bin_centres = 0.5*(hist_stuff[1][1:] + hist_stuff[1][:-1])
# Fill in an expression to evaluate the PDF at the bin_centres.
# To square every element of an array, use **2
pdf = np.exp(-(bin_centres**2.0)/2.0) /(2.0*np.pi)**(1.0/2.0)
bin_width = bin_centres[1] - bin_centres[0]
predicted_bin_heights = pdf * N * bin_width # pdf needs scaling correctly
# Finally, plot the theoretical prediction over the histogram:
plt.plot(bin_centres, predicted_bin_heights, '-r')
plt.show()