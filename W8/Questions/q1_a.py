import numpy as np
import matplotlib.pyplot as plt
from w8_support import *

data = np.load('ct_data.npz')
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']

alpha = 10.0

W_train_fit, b_train_fit = fit_linreg_gradopt(X_train, y_train, alpha)
c_sq_train, [W_bar_train, b_bar_train] = linreg_cost((W_train_fit, b_train_fit), X_train, y_train, alpha)
c_rmse_train = np.sqrt(c_sq_train/len(X_train))

print('Training RMSE: ' + str(c_rmse_train))

c_sq_val, [W_bar_val, b_bar_val] = linreg_cost((W_train_fit, b_train_fit), X_val, y_val, alpha)
c_rmse_val = np.sqrt(c_sq_val/len(X_val))

print('Validation RMSE: ' + str(c_rmse_val))