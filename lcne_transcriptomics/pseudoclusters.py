from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap


def get_arbitrary_orders(adata_sc):
    '''
    for now we will use 2d umap and fit a line. (todo: fit 1d umap and use that)
    this doesn not return the order but some number (that is further used to order the cells)
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
    -----------
    adata_sc : AnnData object
        Input single-cell data with PCA coordinates
    u: the arbitrary numbers that are used to order the cells. potentially this is 1D umap number 
    n_dims : int, default=2
        Number of PCA dimensions to use
    deg : int, default=3 (shoudl not be changed..)
        Degree of polynomial fit
    --------
    dict
        Contains trajectory information:
        - 'order': indices of cells in trajectory order
        - 't_proj': projection coordinates
        - 'polynomials': tuple of polynomial coefficients 
    """
    pca_data = adata_sc.obsm['X_pca'][:,:n_dims]
    u = get_arbitrary_orders(adata_sc)  # u is some arbitrary vlaues (not the order)
    allcells_new_order = np.argsort(u) # order this arbitrary values
    data = pca_data[allcells_new_order] # re-rodered PCA 
    t_proj = u[allcells_new_order]  # re-ordered arbitrary values. for now. not hte final output score. 
   
    # fit separate polynomials for each dimension on t
    polynomials = []
    for dim in range(n_dims):
        p = np.polyfit(t_proj, data[:,dim], deg) 
        polynomials.append(p)
    
    return {
        'data': data,
        'order': allcells_new_order,
        't_proj': t_proj,
        'polynomials': polynomials,
        'n_dims': n_dims
    }


def calculate_projection_scores(trajectory_info, n_points=1000, use_optimizer=False):
    """
    Calculates projection scores and fitted curve coordinates, optionally
    optimizing each projection with a continuous minimizer.
    ----------Parameters
    trajectory_info : dict
        Output from fit_trajectory_curve
    n_points : int
        Number of points for dense sampling
    use_optimizer : bool
        If True, use minimize_scalar per point; otherwise do grid‐search.

    ------- return
    dict with keys
      - 'scores': array (N,) of normalized [0,1] positions
      - 'fitted_curve': list of arrays with coordinates for each dimension
      - 't_opt': array (N,) of optimal t's for each point
      - 't_grid': the t_lin grid used for sampling
      - 'min_d2': minimum squared distances
    """
    data = trajectory_info['data']
    t_proj = trajectory_info['t_proj']
    polynomials = trajectory_info['polynomials']
    n_dims = trajectory_info.get('n_dims', len(polynomials))
    
    t_min, t_max = t_proj.min(), t_proj.max() # still this is some arbitrary values.
    t_lin = np.linspace(t_min, t_max, n_points) # split into many points in this range. 
    
    # Create fitted curve points for each dimension
    fitted_curve = []
    for p in polynomials:
        fitted_curve.append(np.polyval(p, t_lin))
    
    # prepare polynomial evaluation functions for each dimension
    def dim_func(t, dim):
        return np.polyval(polynomials[dim], t)
    
    N = data.shape[0]
    curve_pts = np.vstack(fitted_curve).T        # shape (n_points, 2)
    # compute squared distances between each cell and each sampled curve-point
    d2 = ((data[:, None, :] - curve_pts[None, :, :])**2).sum(axis=2)   # size is N x neighbors
    best_idx = np.argmin(d2, axis=1)             # for each cell, index into t_lin
    t_opt    = t_lin[best_idx]
    scores_0_1   = (t_opt - t_min) / (t_max - t_min)

    return {
        'scores': scores_0_1,
        'fitted_curve': fitted_curve,
        't_opt': t_opt,
        't_grid': t_lin,
        'min_d2': np.min(d2,axis = 1)
    }


def plot_trajectory_3d(trajectory, projections, adata=None, figsize=(10, 8), elev=30, azim=45):
    """
    Creates an interactive 3D plot of trajectory data with the first 3 PCs.
    
    Parameters
    ----------
    trajectory : dict
        Output from fit_trajectory_curve
    projections : dict
        Output from calculate_projection_scores
    adata : AnnData, optional
        Original AnnData object. If provided, uses this instead of trajectory data
    figsize : tuple, default=(10, 8)
        Figure size (width, height)
    elev : float, default=30
        Elevation angle in degrees (vertical rotation)
    azim : float, default=45
        Azimuth angle in degrees (horizontal rotation)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the 3D plot
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np

    if adata is not None:
        data = adata.obsm['X_pca'][trajectory['order'], :3]
    else:
        data = trajectory['data'][:, :3]
    
    scores       = projections['scores']
    fitted_curve = projections['fitted_curve']
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')
    
    # Scatter (low zorder)
    scatter = ax.scatter(
        data[:,0], data[:,1], data[:,2],
        c=scores, cmap='PiYG',
        s=30, edgecolor='k', linewidth=0.2,
        alpha=0.7, zorder=1
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.2)
    cbar.set_label('Pseudocluster', fontsize=14)
    
    # Trajectory line (high zorder)
    if len(fitted_curve) >= 3:
        line = ax.plot(
            fitted_curve[0], fitted_curve[1], fitted_curve[2],
            'r-', linewidth=5, label='Fitted trajectory',
            zorder=10
        )[0]
        # line.set_depthshade(False)
    
    # Remove ticks
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    
    # Bigger labels + extra pad on z
    ax.set_xlabel('PC1', fontsize=16, labelpad=2)
    ax.set_ylabel('PC2', fontsize=16, labelpad=2)
    ax.set_zlabel('PC3', fontsize=16, labelpad=1)
    ax.set_title('Pseudoclusters in PCA (3d)', fontsize=18)
    
    # Equal axis scaling
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        max_range = np.ptp(data, axis=0).max() / 2.0
        mid = data.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.grid(True)
    ax.view_init(elev=elev, azim=azim)

    # tighten but leave 5% on the right for the z-label
    plt.tight_layout(rect=[0, 0, 0.95, 1.0])
    return fig