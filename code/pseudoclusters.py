from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap


def get_arbitrary_orders(adata_sc):
    '''
    for now we will use 2d umap and fit a line. 
    '''
    umapdata = adata_sc.obsm['X_umap'][:,:2]
    Xc = umapdata - umapdata.mean(axis=0)   # centered data
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    v = Vt[0]   # unit‐length 2‐vector
    u = Xc.dot(v)     # alpha is projections along the axis
    return(u)
    
def fit_trajectory_curve(adata_sc, deg=3, n_dims=2):
    """
    Fits a polynomial curve to the PCA data using Isomap for dimensionality reduction.
    
    Parameters:
    -----------
    adata_sc : AnnData object
        Input single-cell data with PCA coordinates
    u: the arbitrary numbers that are used to order the cells. potentially this is 1D umap number 
    n_dims : int, default=2
        Number of PCA dimensions to use
    deg : int, default=3 (shoudl not be changed..)
        Degree of polynomial fit
        
    Returns:
    --------
    dict
        Contains trajectory information:
        - 'order': indices of cells in trajectory order
        - 't_proj': projection coordinates
        - 'polynomials': tuple of polynomial coefficients (px, py)
    """
    pca_data = adata_sc.obsm['X_pca'][:,:n_dims]
    u = get_arbitrary_orders(adata_sc)
    allcells_new_order = np.argsort(u)
    data = pca_data[allcells_new_order]
    t_proj = u[allcells_new_order]
    
    # fit separate polynomials x(t), y(t)    
    px = np.polyfit(t_proj, data[:,0], deg)
    py = np.polyfit(t_proj, data[:,1], deg)
    
    return {
        'data': data,
        'order': allcells_new_order,
        't_proj': t_proj,
        'polynomials': (px, py)
    }



def calculate_projection_scores(trajectory_info, n_points=1000, use_optimizer=False):
    """
    Calculates projection scores and fitted curve coordinates, optionally
    optimizing each projection with a continuous minimizer.
    
    Parameters
    ----------
    trajectory_info : dict
        Output from fit_trajectory_curve, must contain
          - 'data': shape (N,2) array of (x,y)
          - 't_proj': shape (N,) original pseudotimes
          - 'polynomials': tuple(px, py) of poly coefficients
    n_points : int
        Number of points for dense sampling (if use_optimizer=False)
    use_optimizer : bool
        If True, use minimize_scalar per point; otherwise do grid‐search.
    
    Returns
    -------
    dict with keys
      - 'scores': array (N,) of normalized [0,1] positions
      - 'fitted_curve': (x_poly, y_poly) arrays of the dense curve
      - 't_opt': array (N,) of optimal t’s for each point
      - 't_grid': the t_lin grid used for sampling
    """
    data   = trajectory_info['data']
    t_proj = trajectory_info['t_proj']
    px, py = trajectory_info['polynomials']
    
    # dense sampling for plotting (and grid-based fallback)
    t_min, t_max = t_proj.min(), t_proj.max()
    t_lin = np.linspace(t_min, t_max, n_points)
    x_poly = np.polyval(px, t_lin)
    y_poly = np.polyval(py, t_lin)
    
    # prepare polynomial evaluation functions
    def xfun(t): return np.polyval(px, t)
    def yfun(t): return np.polyval(py, t)
    
    N = data.shape[0]
    t_opt = np.empty(N, dtype=float)
    d2_min  = np.empty(N, dtype=float)

    # use bounded scalar minimization for each point
    for i in range(N):
        xi, yi = data[i,0], data[i,1]
        # objective: squared distance
        fun = lambda t: (xfun(t)-xi)**2 + (yfun(t)-yi)**2
        res = minimize_scalar(fun,
#                               bounds=(t_min, t_max),
#                               method='bounded',
                              options={'xatol':1e-4})
        t_opt[i] = res.x
        d2_min[i] = res.fun
    scores_0_1 = (t_opt - t_min) / (t_max - t_min)
    
    return {
        'scores': scores_0_1,
        'fitted_curve': (x_poly, y_poly),
        't_opt': t_opt,
        't_grid': t_lin,
        'min_d2':d2_min
    }
