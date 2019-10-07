import numpy as np


def back_projection(low_obs, high_to_low, sign, search_domain):
    
    if len(low_obs.shape) == 1:
        low_obs = low_obs.reshape((1, low_obs.shape[0]))
    n = low_obs.shape[0]
    high_dim = high_to_low.shape[0]
#    low_dim = low_obs.shape[1]
    high_obs = np.zeros((n, high_dim))
    scale = 1
    for i in range(high_dim):
        high_obs[:, i] = sign[i] * low_obs[:, high_to_low[i]] * scale
    for i in range(n):
        for j in range(high_dim):
            if high_obs[i][j] > bx_size:
                high_obs[i][j] = bx_size
            elif high_obs[i][j] < -bx_size:
                high_obs[i][j] = -bx_size
    return high_obs
