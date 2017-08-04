# -*- coding: utf-8 -*-

import numpy as np
try:
    import cupy as cp
except:
    None
from math import log, pi


class ProbabilityDistribution(object):
    def sampling(self):
        raise NotImplementedError()
    
    def get_param(self):
        raise NotImplementedError()

    def set_param(self):
        raise NotImplementedError()

    def log_likelihood(self):
        raise NotImplementedError()


class MultiVariableGaussian(ProbabilityDistribution):
    def __init__(self, dim, mean=None, var=None, stepsize=None, xp=np):
        self.dim = dim
        self.mean = mean
        self.var = var
        self.stepsize = stepsize
        self.xp = xp
        self.model_class = 'Gaussian'
        
        if self.mean is None:
            self.mean = xp.zeros(dim)
        if self.var is None:
            self.var = xp.identity(dim)
        if self.stepsize is None:
            self.stepsize = 1.
        
        assert self.mean.size == dim and self.var.size == dim * dim, \
            "Invalid value that dimensions DON'T match."
    
    def sampling(self, pop_size):
        sqrt_comat = self.xp.linalg.cholesky(self.var)
        sample = self.mean + np.dot(self.xp.random.normal(0., self.stepsize, (pop_size, self.dim)), sqrt_comat)
        return sample
    
    def get_param(self):
        return self.mean, self.var, self.stepsize
    
    def set_param(self, mean=None, var=None, stepsize=None):
        assert mean.size == dim and var.size == dim * dim, \
            "Invalid value that dimensions DON'T match."
        
        if mean is None:
            self.mean = xp.zeros(dim)
        else:
            self.mean = mean
            
        if var is None:
            self.var = xp.identity(dim)
        else:
            self.var = var
            
        if stepsize is None:
            self.stepsize = 1.
        else:
            self.stepsize = stepsize
    
    def calculate_log_likelihood(self, sample):
        xp = self.xp
        pop_size = sample.shape[0]
        deviation = sample - self.mean
        comats = xp.array([self.stepsize * self.stepsize * self.var for _ in xrange(pop_size)])
        
        try:
            Cinv_der = xp.linalg.solve(comats, deviation)
            log_Cdet = log(xp.linalg.det(self.var))
        except AttributeError:
            Cinv_der = xp.array(np.linalg.solve(comats, deviation))
            log_Cdet = log(np.linalg.det(self.var))
        
        loglikelihood = -0.5 * (self.dim * log(2 * pi) + log_Cdet + xp.diag(xp.dot(Cinv_der, deviation.T)))
        
        return loglikelihood
    
    def use_gpu(self):
        self.xp = cp


# if __name__ == '__main__':
#     x = MultiVariableGaussian(1)
#     s = x.sampling(3)
#     s -= s
#     print s
#     lll = x.calculate_log_likelihood(s)
#     assert np.exp(lll) == 1. / np.sqrt(2 * np.pi),  "Invalid value that likelihood."
