import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap

def fit_trajectory_curve(adata_sc, deg, n_dims=2):
    """
    Fits a polynomial curve to the PCA data using Isomap for dimensionality reduction.
    
    Parameters:
    -----------
    adata_sc : AnnData object
        Input single-cell data with PCA coordinates
    n_dims : int, default=2
        Number of PCA dimensions to use
    deg : int, default=15
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
    isomap = Isomap(n_components=1)
    u = isomap.fit_transform(pca_data).flatten()
    
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

def calculate_projection_scores(trajectory_info, n_points=1000):
    """
    Calculates projection scores and fitted curve coordinates.
    
    Parameters:
    -----------
    trajectory_info : dict
        Output from fit_trajectory_curve
    n_points : int, default=1000
        Number of points for trajectory interpolation
        
    Returns:
    --------
    dict
        Contains projection results:
        - 'scores': normalized projection scores (0-1)
        - 'fitted_curve': tuple of (x,y) coordinates of fitted curve
        - 'distances': squared distances matrix
        - 't_grid': interpolated time points
    """
    data = trajectory_info['data']
    t_proj = trajectory_info['t_proj']
    px, py = trajectory_info['polynomials']
    
    t_lin = np.linspace(t_proj.min(), t_proj.max(), n_points)
    x_poly = np.polyval(px, t_lin)
    y_poly = np.polyval(py, t_lin)
    
    # Calculate distances and projection scores
    # d2 = (x_poly[:,None] - x_poly[None,:])**2 + (y_poly[:,None] - y_poly[None,:])**2
    d2 = (x_poly[:,None] - data[:,0][None,:])**2 + (y_poly[:,None] - data[:,1][None,:])**2
    best_idx = np.argmin(d2, axis=0)
    t_proj_grid = t_lin[best_idx]
    scores_0_1 = (t_proj_grid - t_lin.min()) / (t_lin.max() - t_lin.min())
    
    return {
        'scores': scores_0_1,
        'fitted_curve': (x_poly, y_poly),
        'distances': d2,
        't_grid': t_lin
    }