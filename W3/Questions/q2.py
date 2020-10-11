import numpy as np
import matplotlib.pyplot as plt

def phi_linear(Xin):
    return np.hstack([np.ones((Xin.shape[0],1)), Xin])

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

'''
2a
'''
# C_range = np.arange(1, 21, 1)
# mse_train = np.zeros((len(C_range)))
# mse_val = np.zeros((len(C_range)))

# for i in range(len(C_range)):
#     v_fit = np.linalg.lstsq(X_shuff_train[:, -C_range[i]:], Y_shuff_train[:,None], rcond=None)[0]
#     mse_train[i] = np.mean((np.dot(X_shuff_train[:, -C_range[i]:], v_fit) - Y_shuff_train[:,None])**2.0)
#     mse_val[i] = np.mean((np.dot(X_shuff_val[:, -C_range[i]:], v_fit) - Y_shuff_val[:,None])**2.0)
    
# print('Min MSE training = '+str(np.min(mse_train))+' for C='+str(C_range[np.where(mse_train == np.min(mse_train))]))
# print('Min MSE validation = '+str(np.min(mse_val))+' for C='+str(C_range[np.where(mse_val == np.min(mse_val))]))

'''
2b
'''  
C = 16
v_fit = np.linalg.lstsq(X_shuff_train[:, -C:], Y_shuff_train[:,None], rcond=None)[0]
mse_test = np.mean((np.dot(X_shuff_test[:, -C:], v_fit) - Y_shuff_test[:,None])**2.0)
print('Min MSE test = '+str(mse_test)+' for C='+str(C))
