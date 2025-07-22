import numpy as np
from sklearn.neighbors import NearestNeighbors
import processing
import pandas as pd 


def compute_baseline_distance_between(X1, X2, n_pairs=3000, random_state=0):
    np.random.seed(random_state)
    n1 = X1.shape[0]  # X_sc
    n2 = X2.shape[0]  # X_mer
    idx1 = np.random.choice(n1, n_pairs)
    idx2 = np.random.choice(n2, n_pairs)
    d_rand = np.linalg.norm(X1[idx1] - X2[idx2], axis=1) # size (n_pairs,)
#     distances = np.linalg.norm(X1[idx1] - X2[idx2], axis=1) # distance amongst 1000 pairs. 
    return np.median(d_rand),d_rand  


def compute_null_distribution_for_cell(X_ref, x_query, n_pairs=1000, random_state=None):
    """
    Generate a null distribution of distances between a single query cell
    and random cells from the reference.

    Parameters:
    - X_ref: array-like, shape (n_ref_cells, n_features)
    - x_query: array-like, shape (n_features,) or (1, n_features)
    - n_pairs: int, number of random draws
    - random_state: int or None

    Returns:
    - d_rand: np.ndarray, shape (n_pairs,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    n_ref = X_ref.shape[0]
    # sample reference indices with replacement
    idx = np.random.choice(n_ref, size=n_pairs, replace=True)
    ref_samples = X_ref[idx]
    # compute Euclidean distance to the single query
    # ensure x_query is 1D
    xq = x_query.reshape(1, -1) if x_query.ndim > 1 else x_query
    # if 2D, flatten
    if xq.ndim == 2 and xq.shape[0] == 1:
        xq = xq[0]
    d_rand = np.linalg.norm(ref_samples - xq, axis=1)
    return d_rand



# def impute_pseudocluster(adata_query, adata_ref, k=10, epsilon=1e-10):
def impute_pseudocluster(adata_query, adata_ref, k=10,  epsilon=1e-10, per_cell_null = False):    
    arc_path = '/home/shuonan.chen/scratch_shuonan/scripts/LC_NE_dataintegration/snRNAseq_only/cellID_pc.csv'
    arcinfo = pd.read_csv(arc_path)
    (adata_ref.obs.index == arcinfo['cellID']).all()
    
    common_genes = np.intersect1d(adata_query.var_names, adata_ref.var_names)
    print(f'using {len(common_genes)} genes to run the imputations!')
    adata_ref = adata_ref[:, common_genes].copy()  # we will impute based on this guy, number of `union_genes=2184` will be imputed. 
    
    X_ref = adata_ref[:, common_genes].X
    X_q = adata_query[:, common_genes].X     
#     X_ref_norm = processing.normalize_cols(X_ref, ranked=True)
#     X_q_norm = processing.normalize_cols(X_q, ranked=True)
    X_ref_norm = processing.rankrows(X_ref,standardize = True)
    X_q_norm = processing.rankrows(X_q,standardize = True)
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_ref_norm)
    distances, indices = nbrs.kneighbors(X_q_norm) # distance: size (N_query, K)
    weights = 1 / (distances + epsilon)
    weights = weights / weights.sum(axis=1, keepdims=True)  # normalize so that weights sum to 1
    
    imputed_t_loc = np.zeros((adata_query.n_obs))
    for i in range(adata_query.n_obs):
        neighbor_idx = indices[i]  # length K        
        imputed_t_loc[i] = np.dot(weights[i], arcinfo['pseudoclusters'][neighbor_idx])
        
    baseline_dist,d_rand = compute_baseline_distance_between(X_ref_norm, X_q_norm, n_pairs=1000, random_state=42)        
    
    mean_nn_distances = distances.mean(axis=1)

    if per_cell_null:
    # now generate one null distribution per query cell
        baseline_dists = np.zeros(adata_query.n_obs)
        pscores = np.zeros(adata_query.n_obs)
        for i in range(adata_query.n_obs):
            d_rand_i = compute_null_distribution_for_cell(
                X_ref_norm, X_q_norm[i], random_state=42 + i
            )
            baseline_dists[i] = np.median(d_rand_i)
            # one-sided p-value: fraction of null distances greater than observed mean
            pscores[i] = np.mean(d_rand_i > mean_nn_distances[i])

        confidence_scores = 1 - (mean_nn_distances / baseline_dists)
        confidence_vals = (distances < baseline_dists[:, None]).sum(axis=1)
        frac_vals = confidence_vals / float(k)
    else:    ##  group lelvel    
        mean_nn_distances = np.mean(distances,axis=1)  # size (N_query,)
        confidence_scores = 1 - (mean_nn_distances / baseline_dist)
        confidence_vals = np.sum(distances<baseline_dist, 1)  
        frac_vals = confidence_vals / k
        pscores = (d_rand[:, None] > mean_nn_distances[None, :]).mean(axis=0) # the larger the better!
    conf_dict = {
        'confidence_scores': confidence_scores,
        'confidence_vals': confidence_vals,
        'frac_vals': frac_vals,
        'pscores': pscores
    }
    return imputed_t_loc, conf_dict

