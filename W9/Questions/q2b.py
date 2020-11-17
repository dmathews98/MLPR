import numpy as np
import matplotlib.pyplot as plt
from w9_support import *

data = np.load('ct_data.npz')
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']

alpha = 10.0

def log_sig(aa):
    return 1.0/(1.0 + np.exp(-aa))

def fit_logreg_gradopt(X, yy, alpha):
    D = X.shape[1]
    args = (X, yy, alpha)
    init = (np.zeros(D), np.array(0))
    ww, bb = minimize_list(logreg_cost, init, args)
    return ww, bb

def fit_nn_gradopt(X, yy, alpha, init):
    args = (X, yy, alpha)
    ww, bb, V, bk = minimize_list(nn_cost, init, args)
    return ww, bb, V, bk

# --- TRAINING --- #

# --- W8 initialisation --- #
K = 10    # number of thresholded classification problems to fit
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

# Fit linear regression with regularization of 10 and evaluate the loss
W_train_fit_post, b_train_fit_post = fit_linreg_gradopt(X_train_post, y_train, alpha)
# --- End W8 init --- #

# set W8 fits as init params
init = (W_train_fit_post, b_train_fit_post, W_train_fit, b_train_fit)

# Fit using gradopt
params_fit = fit_nn_gradopt(X_train, y_train, alpha, init)

# Cost evaluation of w8 initialising fit on training set, with regularisation of 10
E, params_bar = nn_cost(params_fit, X_train, y_train, 0)
rmse_train = np.sqrt(E/float(len(X_train)))
print("RMSE training: "+str(rmse_train))