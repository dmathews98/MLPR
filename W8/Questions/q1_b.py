import numpy as np
import matplotlib.pyplot as plt
from w8_support import *

def log_sig(aa):
    return 1.0/(1.0 + np.exp(-aa))

data = np.load('ct_data.npz')
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']

alpha = 10.0

K = 10 # number of thresholded classification problems to fit
mx = np.max(y_train)
mn = np.min(y_train)
hh = (mx-mn)/(K+1)
thresholds = np.linspace(mn+hh, mx-hh, num=K, endpoint=True)
W_train_fit = np.empty((K,X_train.shape[1]))
b_train_fit = np.empty((K))
for kk in range(K):
    labels = y_train > thresholds[kk]
    # ... fit logistic regression to these labels
    W_train_fit[kk], b_train_fit[kk] = fit_logreg_gradopt(X_train, labels, alpha)

# Eval fit to give marginals
X_train_post = log_sig(X_train @ W_train_fit.T + b_train_fit)
X_val_post = log_sig(X_val @ W_train_fit.T + b_train_fit)

# Normalise to make probabilities
# X_train_post = X_train_post/X_train_post.sum(axis=1, keepdims=True)
# X_val_post = X_val_post/X_val_post.sum(axis=1, keepdims=True)

# Fit linear regression with regularization of 10 and evaluate the loss
W_train_fit_post, b_train_fit_post = fit_linreg_gradopt(X_train_post, y_train, alpha)
c_sq_train, [W_bar_train_post, b_bar_train_post] = linreg_cost((W_train_fit_post, b_train_fit_post), X_train_post, y_train, 0)
c_rmse_train = np.sqrt(c_sq_train/len(X_train_post))

print('Training RMSE: ' + str(c_rmse_train))

# evaluate loss of trained model on validation set
c_sq_val, [W_bar_val_post, b_bar_val_post] = linreg_cost((W_train_fit_post, b_train_fit_post), X_val_post, y_val, 0)
c_rmse_val = np.sqrt(c_sq_val/len(X_val_post))

print('Validation RMSE: ' + str(c_rmse_val))