import numpy as np
import matplotlib.pyplot as plt

def phi_linear(Xin):
    return np.hstack([np.ones((Xin.shape[0],1)), Xin])

def phi_quartic(Xin):
    return np.hstack([np.ones((Xin.shape[0],1)), Xin, Xin**2, Xin**3, Xin**4])

def fit_and_plot(phi_fn, T, X):
    w_fit = np.linalg.lstsq(phi_fn(T), X, rcond=None)[0]

    X_grid = np.tile(np.arange(0, 1.01, 0.01)[:,None], (1,1))
    f_grid = np.dot(phi_fn(X_grid), w_fit)
    plt.plot(X_grid, f_grid, linewidth=1)
    # plt.plot([1.0], f_grid[len(f_grid)-1, 0], '.')

def Phi(C, K):
    tt = np.arange(0, 1, 1/20.0)[-C:,None]
    return np.concatenate([tt**i for i in range(K)], axis=1)

def make_vv(C, K):
    return np.dot(Phi(C,K) @ np.linalg.inv(Phi(C,K).T @ Phi(C,K)), np.ones((K, 1)))

np.random.seed(0)

amp_data = np.load('amp_data.npz')['amp_data']

'''
1 a
'''
# plt.figure('Line_graph')
# plt.xlabel('Time /arb')
# plt.ylabel('Amplitude /arb')
# plt.plot(amp_data, 'r')

# plt.figure('Hist_graph')
# plt.xlabel('Amplitude /arb')
# plt.ylabel('Freq')
# plt.hist(amp_data, bins=1000)

'''
1 b
'''
dis = np.shape(amp_data)[0] % 21
C = int(np.shape(amp_data)[0]/21)

cut_data = amp_data[0:np.shape(amp_data)[0]-dis]
wrap_data = np.reshape(cut_data, (C, 21))
shuffle_data = np.random.permutation(wrap_data)

train_share = int(0.7*C)
val_share = int(0.15*C)

X_shuff_train = np.copy(shuffle_data[:train_share, :20])
Y_shuff_train = np.copy(shuffle_data[:train_share, 20])

X_shuff_val = np.copy(shuffle_data[train_share:train_share+val_share, :20])
Y_shuff_val = np.copy(shuffle_data[train_share:train_share+val_share, 20])

X_shuff_test = np.copy(shuffle_data[train_share+val_share:, :20])
Y_shuff_test = np.copy(shuffle_data[train_share+val_share:, 20])

X_shuff_train_copy = np.copy(X_shuff_train)
Y_shuff_train_copy = np.copy(Y_shuff_train)
X_shuff_val_copy = np.copy(X_shuff_val)
Y_shuff_val_copy = np.copy(Y_shuff_val)
X_shuff_test_copy = np.copy(X_shuff_test)
Y_shuff_test_copy = np.copy(Y_shuff_test)

'''
2 a
'''
# T = np.linspace(0, 1, 20, endpoint=False)
# row_index = 15000

# plt.figure('2a')
# plt.plot(T, X_shuff_train[row_index], 'y.')

# fit_and_plot(phi_linear, T[:,None], X_shuff_train[row_index][:,None])
# fit_and_plot(phi_quartic, T[:,None], X_shuff_train[row_index][:,None])

# '''
# 1biii W3
# '''
# plt.plot([1.0], np.dot(make_vv(20, 2).T, X_shuff_train[row_index][:,None])[0], '.')
# plt.plot([1.0], np.dot(make_vv(20, 5).T, X_shuff_train[row_index][:,None])[0], '.')

# plt.show()

'''
W3 1ci
'''
# C_range = np.arange(1, X_shuff_train.shape[1]+1, 1)
# K_range = np.arange(2, 8, 1)
# sq_error = np.empty((len(C_range),len(K_range)))

# for i in range(len(C_range)):
#     for j in range(len(K_range)):
#         sq_error[i, j] = np.mean((np.dot(make_vv(C_range[i],K_range[j]).T, X_shuff_train.T[-C_range[i]:,]) - Y_shuff_train.T)**2.0)

# print('Min square error was ' + str(np.min(sq_error)))
# print('C val is ' + str(C_range[np.where(sq_error == np.min(sq_error))[0]]))
# print('K val is ' + str(K_range[np.where(sq_error == np.min(sq_error))[1]]))

'''
W3 1cii
'''
C = 2
K = 2
vv = make_vv(C, K)

mse_train = np.mean(((vv.T @ X_shuff_train.T[-C:,]) - Y_shuff_train.T)**2.0)
mse_val = np.mean(((vv.T @ X_shuff_val.T[-C:,]) - Y_shuff_val.T)**2.0)
mse_test = np.mean(((vv.T @ X_shuff_test.T[-C:,]) - Y_shuff_test.T)**2.0)

print('MSE for training set C='+str(C)+' and K='+str(K)+' is '+str(mse_train))
print('MSE for validation set C='+str(C)+' and K='+str(K)+' is '+str(mse_val))
print('MSE for test set C='+str(C)+' and K='+str(K)+' is '+str(mse_test))
