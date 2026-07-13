import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev,splrep

def get_derivatives(x,y, visualize = False):
    
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    interior_knots = np.linspace(x_sorted.min(),x_sorted.max(),5)[1:-1]
    tck = splrep(x_sorted, y_sorted,k=3,t=interior_knots)
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = splev(x_fit, tck)                 # function value
    dy_dx = splev(x_fit, tck, der=1)          # first derivative
    d2y_dx2 = splev(x_fit, tck, der=2)        # second derivative
#     check_idx = np.where((x_fit>0) &(x_fit<1))[0]
    eps = 0.15  # 5% margin; tune
    check_idx = np.where((x_fit > eps) & (x_fit < 1 - eps))[0]

    
    y_pred = splev(x_sorted, tck)
    resid = y_sorted - y_pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_sorted - np.mean(y_sorted))**2)
    r2 = 1 - ss_res / ss_tot

    if visualize:
        plt.figure(figsize=(4, 5))
        plt.scatter(x, y, s=1, alpha=0.3, label='raw data')
        plt.plot(x_fit, y_fit, color='black', label='spline fit')
        plt.plot(x_fit, dy_dx, color='blue', label='1st derivative')
        plt.legend()
        plt.show()
    return(dy_dx[check_idx], d2y_dx2[check_idx],x_fit[check_idx], y_fit[check_idx],r2)
#     return(dy_dx, d2y_dx2,x_fit, y_fit)


def find_u_shaped_genes(expr_ordered, gene_names, x,
                        slope_lo=-8, slope_hi=8,
                        d1_thresh=0, d2_thresh=0,
                        return_diagnostics=False):
    """Identify genes whose expression is U-shaped along a pseudotime coordinate.

    A gene qualifies when its spline fit vs ``x`` has a U-shaped first derivative that
    flips from steeply negative to steeply positive across the interior window (expression
    goes down then up): ``dy_dx[0] < slope_lo`` and ``dy_dx[-1] > slope_hi`` (with opposite
    signs). This programmatically derives the ``u_genes`` list used downstream.

    Note on lineage: the original derivation (see
    ``notebooks/outdated/03_check_gradient*.ipynb``) prepended a "sudden-drop" gate
    (``max|dy/dx| > 15`` and ``max|d2y/dx2| > 100``). That gate is retained here as the
    optional ``d1_thresh``/``d2_thresh`` knobs but defaults to OFF, because on the current
    snRNA pipeline (BN + z-scale, 10-dim trajectory) it biases toward sharp asymmetric
    drops and excludes genuine symmetric U-shapes. The U-shape slope criterion alone is the
    primary signal.

    Parameters
    ----------
    expr_ordered : (n_cells, n_genes) array
        Scaled expression (e.g. ``adata_sc.X``) reordered by trajectory order and aligned
        to ``x``. Thresholds assume z-scaled expression (``sc.pp.scale``).
    gene_names : sequence of str, length n_genes  (e.g. ``adata_sc.var_names``)
    x : (n_cells,) array
        Pseudocluster score (``scores_0_1``), aligned to ``expr_ordered`` rows.
    slope_lo, slope_hi : float
        U-shape thresholds on the first derivative at the ends of the interior window:
        require ``dy_dx[0] < slope_lo`` and ``dy_dx[-1] > slope_hi`` (with opposite signs).
    d1_thresh, d2_thresh : float
        Optional sudden-drop gate on ``max|dy/dx|`` and ``max|d2y/dx2|`` over the interior
        window (0 disables, the default).
    return_diagnostics : bool
        If True, return a dict with per-gene diagnostics instead of just the gene list.

    Returns
    -------
    list[str]  (or dict with per-gene diagnostics if ``return_diagnostics``)
    """
    expr_ordered = np.asarray(expr_ordered)
    gene_names   = np.asarray(gene_names)
    n_genes      = expr_ordered.shape[1]

    max_d1  = np.empty(n_genes)
    max_d2  = np.empty(n_genes)
    d1_ends = np.empty((n_genes, 2))
    for g in range(n_genes):
        dy_dx, d2y_dx2, _, _, _ = get_derivatives(x, expr_ordered[:, g])
        max_d1[g]  = np.max(np.abs(dy_dx))
        max_d2[g]  = np.max(np.abs(d2y_dx2))
        d1_ends[g] = (dy_dx[0], dy_dx[-1])

    sudden  = (max_d1 > d1_thresh) & (max_d2 > d2_thresh)
    u_shape = ((d1_ends[:, 0] * d1_ends[:, 1] < 0) &
               (d1_ends[:, 0] < slope_lo) &
               (d1_ends[:, 1] > slope_hi))
    keep  = np.where(sudden & u_shape)[0]
    genes = list(gene_names[keep])

    if return_diagnostics:
        return {"genes": genes, "sudden_idx": np.where(sudden)[0],
                "max_d1": max_d1, "max_d2": max_d2, "d1_ends": d1_ends}
    return genes

