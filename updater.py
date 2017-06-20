import numpy as np
import collections
from math import log, pi


class Updater(object):
    """Base class of all updater."""
    def setup(self, distribution, weight_function, lr):
        self.target = distribution
        self.t = 0
        self.w_func = weight_function
        self.lr = lr
        
        if not callable(self.w_func):
            raise TypeError('weight function is NOT callable.')

    def update(self):
        raise NotImplementedError()


class NaturalGradientUpdater(Updater):
    def __init__(self):
        super(Updater, self).__init__()
    
    def update(self, fitness, sample):
        self.t += 1
        
        weight = self.w_func(fitness)
        
        if self.target.model_class in 'Gaussian':
            self.gaussian_param_update(weight, sample)
        if self.target.model_class in 'Bernoulli':
            self.bernouil_param_update(weight, sample)
    
    def gaussian_param_update(self, weight, sample):
        #TODO:Implementaion of Comulative Step-size Update
        #TODO:Implementaion of rank-one Update
        
        mean, var = self.target.get_param()
        grad_m, grad_var = self._compute_natural_grad_gaussian(weight, sample, mean, var)
       
        new_mean = mean + self.lr['mean'] * grad_m
        new_var = var + self.lr['var'] * grad_var
        
        self.target.set_param(new_mean, new_var)
    
    def bernouil_param_update(self, weight, sample):
        # TODO:Implementaion of Population-based increment learning
        pass

    def _compute_natural_grad_gaussian(self, weight, sample, mean, var):
        derivation = sample - mean
        w_der = weight * derivation.T
        grad_m = w_der.sum(axis=1)

        if self.target.model_class in 'Isotropic':
            norm_w_der = xp.diag(xp.dot(w_der, w_der.T))
            grad_var = (xp.sum(weight * norm_w_der) / self.target.dim) - (xp.sum(weight) * var)
        elif self.target.model_class in 'Separable':
            grad_var = (w_der * derivation.T).sum(axis=1) - (weights.sum() * var)
        else:
            grad_var = xp.dot(w_der, derivation) - (weight.sum() * var)
        
        return grad_m, grad_var