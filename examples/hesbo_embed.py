import numpy as np
import copy

class projection:
    def __init__(self, low_dim, obj_func):
        # attributes of the original function
        self._dim = low_dim
        self._search_domain = copy.deepcopy(obj_func._search_domain[:low_dim])
        self._num_init_pts = obj_func._num_init_pts
        self._sample_var = obj_func._sample_var
        self._min_value = obj_func._min_value
        self._observations = obj_func._observations
        self._num_fidelity = obj_func._num_fidelity
        
        # new attributes
        self.obj_func = obj_func
        self._org_search_domain = obj_func._search_domain
        self._high_to_low = np.random.choice(range(low_dim), obj_func._dim)
        self._sign = np.random.choice([-1, 1], obj_func._dim)
        
    
    def org_to_box(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))
        for i in range(min(len(self._org_search_domain),x.shape[1])):
            x[:, i] = (x[:,i]-(self._org_search_domain[i, 1] + self._org_search_domain[i, 0]) / 2) * 2 / (
                    self._org_search_domain[i, 1] - self._org_search_domain[i, 0])
        return x
    
    def box_to_org(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))
        for i in range(min(len(self._org_search_domain),x.shape[1])):
            x[:, i] = x[:, i] * (self._org_search_domain[i, 1] - self._org_search_domain[i, 0]) / 2 + (
                        self._org_search_domain[i, 1] + self._org_search_domain[i, 0]) / 2
        return x
    
    def back_projection(self, low_obs):
        
        low_obs = self.org_to_box(low_obs)
        n = low_obs.shape[0]
        high_dim = self._high_to_low.shape[0]
    #    low_dim = low_obs.shape[1]
        high_obs = np.zeros((n, high_dim))
        scale = 1
        for i in range(high_dim):
            high_obs[:, i] = self._sign[i] * low_obs[:, self._high_to_low[i]] * scale
        high_obs = self.box_to_org(high_obs)
        return np.squeeze(high_obs)
    
    def evaluate_true(self, x):
        return self.obj_func.evaluate_true(self.back_projection(x))
        
    def evaluate(self, x):
        return self.obj_func.evaluate(self.back_projection(x))
