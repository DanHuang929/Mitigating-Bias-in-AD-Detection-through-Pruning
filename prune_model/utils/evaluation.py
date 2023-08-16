import numpy as np

import torch

from sklearn.metrics import (average_precision_score, balanced_accuracy_score, accuracy_score)
from sklearn.metrics import roc_auc_score, f1_score


def compute_empirical_bias(y_pred, y_true, priv, attr, prune_with_tnr_info):

    def zero_if_nan(x):
        if isinstance(x, torch.Tensor):
            return 0. if torch.isnan(x) else x
        else:
            return 0. if np.isnan(x) else x
    
    bias = 0
    
    if(attr=="gender" or attr=="age"):
        
        tp_0 = zero_if_nan(y_pred[(1 - priv) * y_true == 1].mean())
        tp_1 = zero_if_nan(y_pred[priv * y_true == 1].mean()) 
        tn_0 = 1-zero_if_nan(y_pred[(1 - priv) * (1 - y_true) == 1].mean())
        tn_1 = 1-zero_if_nan(y_pred[priv * (1 - y_true) == 1].mean()) 
        
        print(tp_0, tp_1)
        print(tn_0, tn_1)
        
        if prune_with_tnr_info==1:
            bias = (tp_0+tn_0) - (tp_1+tn_1)
        else:
            bias = tp_0 - tp_1

    return bias


def get_objective(y_pred, y_true, priv, attr, prune_with_tnr_info):
    """Evaluates the objective function on the provided data"""
    bias = compute_empirical_bias(y_pred, y_true, priv, attr, prune_with_tnr_info)
    performance = balanced_accuracy_score(y_true, y_pred)
#     performance = accuracy_score(y_true, y_pred)
    return {'bias': bias, 'performance': performance}

def get_test_objective_(y_pred, y_test, p_test, attr, prune_with_tnr_info):
    """Returns point-estimate objective function values on the test data"""
    return get_objective(y_pred, y_test, p_test,
                         attr, prune_with_tnr_info)
