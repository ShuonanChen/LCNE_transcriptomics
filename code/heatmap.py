import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns


def downsample_moving_avg(X, n_out=100):
    """
    Downsample by explicit binning along the column (pseudotime) axis.

    X: array of shape (genes x cells), with cells ordered along trajectory
    n_out: desired number of pseudotime bins

    Returns: array of shape (genes x n_out)
    """
    X = np.asarray(X)
    n_rows, n_cols = X.shape

    # cap n_out by number of columns
    n_out = min(n_out, n_cols)
    if n_out < 1:
        raise ValueError("n_out must be >= 1")
    if n_out == n_cols:
        return X.copy()

    # integer bin edges in column index space
    edges = np.linspace(0, n_cols, n_out + 1).astype(int)

    Y = np.zeros((n_rows, n_out), dtype=float)
    for i in range(n_out):
        start, end = edges[i], edges[i + 1]
        if end <= start:
            end = min(start + 1, n_cols)
        Y[:, i] = X[:, start:end].mean(axis=1)

    return Y


def scaling(hm, col_scale=True):
    """
    Row-wise min–max scaling: each gene spans [0, 1] across bins.
    """
    hm_out = hm.copy().astype(float)
    minval = np.min(hm_out, 1)
    hm_out -= minval[:, None]

    maxval = np.max(hm_out, 1)
    # avoid division by zero for flat genes
    maxval[maxval == 0] = 1.0
    hm_out /= maxval[:, None]
    return hm_out


def reorder_genes(reduced):
    """
    Hierarchical clustering on genes.

    reduced: rows = genes, columns = binned samples (G x n_bins)
    returns ordered_data (same shape) and row_order indices.
    """
    Z = linkage(reduced, method='ward', metric='euclidean', optimal_ordering=True)
    dendro = dendrogram(Z, no_plot=True)
    row_order = dendro['leaves']
    ordered_data = reduced[row_order]
    return ordered_data, row_order


def order_genes_by_correlation(reduced, ascending=True):
    """
    Order genes by correlation with pseudotime-like index.

    reduced: G x n_bins (after binning + scaling).
    We treat bin index (0..1) as pseudotime and compute Pearson
    correlation per gene with that axis.

    Returns ordered_data and row_order indices.
    """
    reduced = np.asarray(reduced, dtype=float)
    G, B = reduced.shape
    print('****order by correlation now****')
    # "pseudotime" along bins: 0..1
    t = np.linspace(0.0, 1.0, B)
    t_centered = t - t.mean()
    t_norm = np.linalg.norm(t_centered)
    if t_norm == 0:
        # degenerate case, just return as-is
        row_order = np.arange(G)
        return reduced.copy(), row_order

    # center each gene across bins
    gene_centered = reduced - reduced.mean(axis=1, keepdims=True)
    num = gene_centered @ t_centered
    denom = np.linalg.norm(gene_centered, axis=1) * t_norm
    denom[denom == 0] = 1.0

    corrs = num / denom

    if ascending:
        row_order = np.argsort(corrs)       # strong negative → strong positive
    else:
        row_order = np.argsort(-corrs)

    ordered_data = reduced[row_order]
    return ordered_data, row_order


def get_final_heatmap(inputheatmap,
                      gene_reorder=True,
                      order_mode="cluster",
                      n_out=50):
    """
    inputheatmap:
        G x N matrix (genes x cells), cells already ordered appropriately
        e.g. adata[:, allyourgenes].X[trajectory['order']].copy()[np.argsort(cellscores)].T

    Steps:
        - bin along pseudotime into n_out bins
        - per-gene min–max scaling to [0, 1]
        - reorder genes either by clustering or correlation (optional)

    Returns:
        ordered_data: G x n_out
        row_order: index array mapping back to original genes
    """
    # 1) bin along pseudotime
    reduced = downsample_moving_avg(inputheatmap, n_out=n_out)  # (G, n_out)

    # 2) per-gene [0, 1] scaling
    reduced = scaling(reduced)

    # 3) ordering
    if gene_reorder:
        if order_mode == "cluster":
            ordered_data, row_order = reorder_genes(reduced)
        elif order_mode == "corr":
            ordered_data, row_order = order_genes_by_correlation(reduced, ascending=True)
        else:
            # unknown mode → no reordering
            row_order = np.arange(reduced.shape[0])
            ordered_data = reduced.copy()
    else:
        row_order = np.arange(reduced.shape[0])
        ordered_data = reduced.copy()

    return ordered_data, row_order