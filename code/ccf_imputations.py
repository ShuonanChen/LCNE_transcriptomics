import numpy as np
from sklearn.neighbors import NearestNeighbors
import processing

def impute_spatial_coordinates(adata_query, adata_ref, k=3, epsilon=1e-10, weight_mode="inverse", tau=1.0):
    """
    Impute spatial coordinates for query cells based on reference cells using k-nearest neighbors.
    epsilon     : float -> Small constant
    weight_mode : {"inverse", "softmax"} -> How to convert distances into weights
    tau         : float -> Temperature for softmax
    """
    common_genes = np.intersect1d(adata_ref.var_names, adata_query.var_names)
    X_query = adata_query[:, common_genes].X
    X_ref   = adata_ref[:, common_genes].X
    X_query_norm = processing.rankrows(X_query)
    X_ref_norm   = processing.rankrows(X_ref)

    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_ref_norm)
    distances, indices = nbrs.kneighbors(X_query_norm)

    if weight_mode == "inverse":
        weights = 1 / (distances + epsilon)
        weights = weights / weights.sum(axis=1, keepdims=True)
    elif weight_mode == "softmax":
        z = -distances / tau
        z = z - z.max(axis=1, keepdims=True)  # second term size is N_query (max across neighbors)
        weights = np.exp(z)
        weights = weights / weights.sum(axis=1, keepdims=True)
    else:
        raise ValueError("weight_mode must be 'inverse' or 'softmax'")

    ref_spatial = processing.get_hemi(adata_ref.obsm['spatial'],
                                      meshhome = '/allen/aind/scratch/shuonan.chen/scripts/Pons_MERFISH/mesh/')

    imputed_coords = np.zeros((adata_query.n_obs, 3))
    for i in range(adata_query.n_obs):
        neighbor_idx = indices[i]
        imputed_coords[i] = np.dot(weights[i], ref_spatial[neighbor_idx])

    return imputed_coords

