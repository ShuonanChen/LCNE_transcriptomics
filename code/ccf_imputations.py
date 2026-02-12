import numpy as np
from sklearn.neighbors import NearestNeighbors
import processing, imputations

def impute_spatial_coordinates(adata_query, adata_ref, k=3, epsilon=1e-10, similarity_transform="inverse", tau=1.0):
    """
    Impute spatial coordinates for query cells based on reference cells using k-nearest neighbors.
    epsilon     : float -> Small constant
    similarity_transform : {"inverse", "softmax",'gaussian'} -> How to convert distances into weights
    tau         : float -> Temperature for softmax
    """
    common_genes = np.intersect1d(adata_ref.var_names, adata_query.var_names)
    X_query = adata_query[:, common_genes].X
    X_ref   = adata_ref[:, common_genes].X
    X_query_norm = processing.rankrows(X_query)
    X_ref_norm   = processing.rankrows(X_ref)

    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_ref_norm)
    distances, indices = nbrs.kneighbors(X_query_norm)

    if similarity_transform=='softmax':
        weights = imputations.weight_softmax_dst(distances, tau = 0.1)
    elif similarity_transform=='inverse':
        weights = imputations.weight_inverse_dst(distances,epsilon = 1e-10)
    elif similarity_transform=='gaussian':
        weights = imputations.weight_gaussian_dst(distances)
    else:
        raise ValueError("similarity_transform must be 'inverse' or 'softmax' or 'gaussian'")

    ref_spatial = processing.get_hemi(adata_ref.obsm['spatial'],
                                      meshhome = '/allen/aind/scratch/shuonan.chen/scripts/Pons_MERFISH/mesh/')

    imputed_coords = np.zeros((adata_query.n_obs, 3))
    for i in range(adata_query.n_obs):
        neighbor_idx = indices[i]
        imputed_coords[i] = np.dot(weights[i], ref_spatial[neighbor_idx])

    return imputed_coords

