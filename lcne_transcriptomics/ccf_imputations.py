import numpy as np
from sklearn.neighbors import NearestNeighbors
from . import processing

def impute_spatial_coordinates(adata_query, adata_ref, k=3, epsilon=1e-10):
    """
    Impute spatial coordinates for query cells based on reference cells using k-nearest neighbors.
    
    Parameters
    ----------
    adata_query : AnnData
        Query dataset containing cells to impute coordinates for
    adata_ref : AnnData
        Reference dataset with known spatial coordinates
    k : int, default=3
        Number of nearest neighbors to use
    epsilon : float, default=1e-10
        Small constant to avoid division by zero
        
    Returns
    -------
    np.ndarray
        Imputed 3D coordinates of shape (n_query_cells, 3)
    """
    # Find common genes between query and reference
    common_genes = np.intersect1d(adata_ref.var_names, adata_query.var_names)
    
    # Extract and normalize expression matrices
    X_query = adata_query[:, common_genes].X
    X_ref = adata_ref[:, common_genes].X
#     X_query_norm = processing.normalize_cols(X_query, ranked=True)
#     X_ref_norm = processing.normalize_cols(X_ref, ranked=True)
    X_query_norm = processing.rankrows(X_query)
    X_ref_norm = processing.rankrows(X_ref)
    
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_ref_norm)
    distances, indices = nbrs.kneighbors(X_query_norm)
    
    # Calculate weights
    weights = 1 / (distances + epsilon)
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Get spatial coordinates from reference
    ref_spatial = adata_ref.obsm['spatial']
    ref_spatial = processing.get_hemi(ref_spatial)
    
    # Impute coordinates
    imputed_coords = np.zeros((adata_query.n_obs, 3))
    for i in range(adata_query.n_obs):
        neighbor_idx = indices[i]
        imputed_coords[i] = np.dot(weights[i], ref_spatial[neighbor_idx])
        
    return imputed_coords