import numpy as np
from sklearn.neighbors import NearestNeighbors
import processing

def impute_pseudocluster(adata_query, adata_ref, k=10, epsilon=1e-10):
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
        Imputed pseudocluster score (n_query_cells, 1)
    """
    ## todo: the arc info should really be saved for each cell on snRNAseq instead of the separate file.. prob okay for now.
    import pickle 
    arc_info = pickle.load(open('/home/shuonan.chen/scratch_shuonan/LC_scRNAseq/arc_info.pkl', 'rb'))
    locals().update(arc_info)
    inverse_order = np.argsort(arc_info['sort_indices'])
    t_projections_original_order = arc_info['t_projections'][inverse_order]
    
    # Find common genes between query and reference
    common_genes = np.intersect1d(adata_ref.var_names, adata_query.var_names)
    
    # Extract and normalize expression matrices
    X_query = adata_query[:, common_genes].X
    X_ref = adata_ref[:, common_genes].X
    X_query_norm = processing.normalize_cols(X_query, ranked=True)
    X_ref_norm = processing.normalize_cols(X_ref, ranked=True)
    
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_ref_norm)
    distances, indices = nbrs.kneighbors(X_query_norm)
    
    # Calculate weights
    weights = 1 / (distances + epsilon)
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    imputed_t_loc = np.zeros((adata_query.n_obs))
    for i in range(adata_query.n_obs):
        neighbor_idx = indices[i]    
        imputed_t_loc[i] = np.dot(weights[i], t_projections_original_order[neighbor_idx])

    return imputed_t_loc