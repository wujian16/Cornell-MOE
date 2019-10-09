import numpy as np

def org_to_box(x, search_domain):
    if len(x.shape) == 1:
        x = x.reshape((1, x.shape[0]))
    for i in range(min(len(search_domain),x.shape[1])):
        x[:, i] = (x[:,i]-(search_domain[i, 1] + search_domain[i, 0]) / 2) * 2 / (
                search_domain[i, 1] - search_domain[i, 0])
    return x

def box_to_org(x, search_domain):
    if len(x.shape) == 1:
        x = x.reshape((1, x.shape[0]))
    for i in range(min(len(search_domain),x.shape[1])):
        x[:, i] = x[:, i] * (search_domain[i, 1] - search_domain[i, 0]) / 2 + (
                    search_domain[i, 1] + search_domain[i, 0]) / 2
    return x

def back_projection(low_obs, high_to_low, sign, search_domain):
    
    low_obs = org_to_box(low_obs, search_domain)
    n = low_obs.shape[0]
    high_dim = high_to_low.shape[0]
#    low_dim = low_obs.shape[1]
    high_obs = np.zeros((n, high_dim))
    scale = 1
    for i in range(high_dim):
        high_obs[:, i] = sign[i] * low_obs[:, high_to_low[i]] * scale
    high_obs = box_to_org(high_obs, search_domain)
    return high_obs
