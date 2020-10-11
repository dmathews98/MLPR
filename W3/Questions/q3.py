import numpy as np
import matplotlib.pyplot as plt

def Phi(C, K):
    tt = np.arange(0, 1, 1/20.0)[-C:,None]
    return np.concatenate([tt**i for i in range(K)], axis=1)

def make_vv(C, K):
    return np.dot(Phi(C,K) @ np.linalg.inv(Phi(C,K).T @ Phi(C,K)), np.ones((K, 1)))

np.random.seed(0)

amp_data = np.load('amp_data.npz')['amp_data']

size = 21

dis = np.shape(amp_data)[0] % size
C = int(np.shape(amp_data)[0]/size)

cut_data = amp_data[0:np.shape(amp_data)[0]-dis]
wrap_data = np.reshape(cut_data, (C, size))
shuffle_data = np.random.permutation(wrap_data)

train_share = int(0.7*C)
val_share = int(0.15*C)

X_shuff_train = np.copy(shuffle_data[:train_share, :size-1])
Y_shuff_train = np.copy(shuffle_data[:train_share, size-1])

X_shuff_val = np.copy(shuffle_data[train_share:train_share+val_share, :size-1])
Y_shuff_val = np.copy(shuffle_data[train_share:train_share+val_share, size-1])

X_shuff_test = np.copy(shuffle_data[train_share+val_share:, :size-1])
Y_shuff_test = np.copy(shuffle_data[train_share+val_share:, size-1])

# C_range = np.arange(1, size, 1)
# hh_range = np.arange(0, 11, 1)
# mse_train = np.zeros((len(hh_range), len(C_range)))
# mse_val = np.zeros((len(hh_range), len(C_range)))

# for j in range(len(hh_range)):
#     for i in range(len(C_range)):
#         phi_train = np.vstack([X_shuff_train[:, -C_range[i]:], hh_range[j]**(0.5)*np.identity(C_range[i])])
#         yy_t_train = np.vstack([Y_shuff_train[:,None], np.zeros((C_range[i],1))])

#         v_fit = np.linalg.lstsq(phi_train, yy_t_train, rcond=None)[0]
#         mse_train[j, i] = (np.sum((yy_t_train - np.dot(phi_train, v_fit))**2.0) - hh_range[j]*np.dot(v_fit.T, v_fit)[0,0])/Y_shuff_train.shape[0]

#         phi_val = np.vstack([X_shuff_val[:, -C_range[i]:], hh_range[j]**(0.5)*np.identity(C_range[i])])
#         yy_t_val = np.vstack([Y_shuff_val[:,None], np.zeros((C_range[i],1))])
#         mse_val[j, i] = (np.sum((yy_t_val - np.dot(phi_val, v_fit))**2.0) - hh_range[j]*np.dot(v_fit.T, v_fit)[0,0])/Y_shuff_val.shape[0]

# print('Min MSE training = '+str(np.min(mse_train))+' for C='+str(C_range[np.where(mse_train == np.min(mse_train))[1]])+' and h='+str(hh_range[np.where(mse_train == np.min(mse_train))[0]]))
# print('Min MSE validation = '+str(np.min(mse_val))+' for C='+str(C_range[np.where(mse_val == np.min(mse_val))[1]])+' and h='+str(hh_range[np.where(mse_val == np.min(mse_val))[0]]))

C = 16
hh = 3.5
phi_train = np.vstack([X_shuff_train[:, -C:], hh**(0.5)*np.identity(C)])
yy_t_train = np.vstack([Y_shuff_train[:,None], np.zeros((C,1))])
phi_test = np.vstack([X_shuff_test[:, -C:], hh**(0.5)*np.identity(C)])
yy_t_test = np.vstack([Y_shuff_test[:,None], np.zeros((C,1))])
v_fit = np.linalg.lstsq(phi_train, yy_t_train, rcond=None)[0]
mse_test = (np.sum((yy_t_test - np.dot(phi_test, v_fit))**2.0) - hh*np.dot(v_fit.T, v_fit)[0,0])/Y_shuff_test.shape[0]
print('Min MSE test = '+str(mse_test)+' for C='+str(C))

# T = np.linspace(0, 1, C, endpoint=False)
# row_index = 15000

# plt.clf()
# # plt.plot(T, (Phi(C, 16)).dot(X_shuff_train[row_index, -C:]))
# plt.plot(T, X_shuff_train[row_index, -C:], 'y.')
# plt.show()