import pandas as pd
import anndata
import numpy as np
from scipy.stats import rankdata
import utils


def rankrows(M, standardize = True):
    try:
        result = M.toarray().copy() # a x b
    except:
        result = M.copy()
    result = np.apply_along_axis(rankdata, 1, result) # if axis = 1, ranking for each row / cells        
    if standardize:
        means = np.mean(result, axis=1)
        stds = np.std(result, axis=1, ddof=0)
        stds[stds == 0] = 1e-10
        result = (result - means[:,None]) / stds[:,None]
    return result


def flip(a, xm):
        return(2*xm-a)

    
def get_hemi(S_mer, mesh=None):
    '''
    assume the axis of interest are both on the last axis. 
    '''
    if mesh is None:
        allmeshes = utils.load_sym_mesh()
        mesh = allmeshes[-1]
    xm = np.min(mesh.vertices[:,-1]) + np.ptp(mesh.vertices[:,-1])/2 # this is the center line to indicate the hemisphere 
    new_coords = S_mer.copy()
    new_coords[:,-1] = np.where(new_coords[:,-1] > xm, flip(new_coords[:,-1],xm), new_coords[:,-1])    
    return(new_coords)





def normalize_cols(M, ranked=True):
    """
    not use
    """
    raise RuntimeError("normalize_cols is deprecated - use `rankrows` instead")
    try:
        result = M.toarray().copy() # a x b
    except:
        result = M.copy()
    if ranked:  # output shape: 
        result = np.apply_along_axis(rankdata, 0, result) 
    means = np.mean(result, axis=0)
    stds = np.std(result, axis=0, ddof=0)
    stds[stds == 0] = 1e-10
    result = (result - means) / stds
    return result
