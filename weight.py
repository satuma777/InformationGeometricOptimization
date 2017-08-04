import numpy as np


def cma_like_weight(q_plus, q_minus, xp):
    weight_plus = -2. * xp.log(2. * q_plus)
    weight_plus[xp.where(q_plus < 0.5)] = 0
    
    if q_minus is not None:
        weight_minus = -2. * xp.log(2. * q_minus)
        weight_minus[xp.where(q_minus < 0.5)] = 0
        
        q_plus_truncated = xp.where(q_plus < 0.5, q_plus, 0.5)
        q_minus_truncated = xp.where(q_minus < 0.5, q_minus, 0.5)

        diff_idx = xp.where(q_plus != q_minus)
        
        weight_plus[diff_idx] = (q_plus_truncated[diff_idx] * weight_plus[diff_idx]
                                 - q_minus_truncated[diff_idx] * weight_minus[diff_idx]
                                 + 2. * (q_plus_truncated[diff_idx] + q_minus_truncated[diff_idx]))
        
        weight_plus[diff_idx] /= q_plus[diff_idx] - q_minus[diff_idx]
        
    return weight_plus


class QuantileBasedWeight(object):
    def __init__(self, minimization=True, tie_case=False, non_increasing_function=cma_like_weight, xp=np):
        self.min = minimization
        self.tie_case = tie_case
        self.non_inc_func = non_increasing_function
        self.xp = xp
    
    def __call__(self, evaluation):
        xp = self.xp
        pop_size = len(evaluation)
        q_plus = None
        q_minus = None
        
        if self.min:
            q_plus = xp.array([len(evaluation[xp.where(evaluation <= eval)]) for eval in evaluation])
            print(q_plus)
            q_plus = q_plus / pop_size
            if self.tie_case:
                q_minus = xp.array([len(evaluation[xp.where(evaluation < eval)]) for eval in evaluation])
                q_minus /= pop_size
        
        else:
            q_plus = xp.array([evaluation[xp.where(evaluation >= eval)].size() for eval in evaluation])
            q_plus /= pop_size
            if self.tie_case:
                q_minus = xp.array([evaluation[xp.where(evaluation > eval)].size() for eval in evaluation])
                q_minus /= pop_size
        
        return self.non_inc_func(q_plus, q_minus, self.xp) / pop_size

#TODO: LebesgueMeasureBasedWeight [Akimoto2012(GECCO2012)]