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

# --- TRAINING --- #

# Random initialising of small parameters
K = 10    # Just assumed as 10 classes last week
ww = np.random.randint(low=-100, high=100, size=K)/1000.0
bb = np.random.randint(low=-100, high=100, size=1)/1000.0
V = np.random.randint(low=-100, high=100, size=(K,len(X_train[0])))/1000.0
bk = np.random.randint(low=-100, high=100, size=K)/1000.0

def fit_nn_gradopt(X, yy, alpha, init):
    args = (X, yy, alpha)
    ww, bb, V, bk = minimize_list(nn_cost, init, args)
    return ww, bb, V, bk

# Fit using gradopt
params_fit = fit_nn_gradopt(X_train, y_train, alpha, (ww, bb, V, bk))

# Cost evaluation of random initialising fit on training set, with regularisation of 10
E, params_bar = nn_cost(params_fit, X_train, y_train, 0)
rmse_train = np.sqrt(E/float(len(X_train)))
print("RMSE training: "+str(rmse_train))