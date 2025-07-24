import numpy as np
from sklearn.neighbors import NearestNeighbors
from . import processing
import pandas as pd 


def compute_baseline_distance_between(X1, X2, n_pairs=3000, random_state=0):
    """
    Compute the median and full null distribution of distances between random pairs
    of rows from X1 and X2."""
    np.random.seed(random_state)
    n1, n2 = X1.shape[0], X2.shape[0]
    idx1 = np.random.choice(n1, size=n_pairs, replace=True)
    idx2 = np.random.choice(n2, size=n_pairs, replace=True)
    d_rand = np.linalg.norm(X1[idx1] - X2[idx2], axis=1)
    return np.median(d_rand), d_rand


def compute_null_distribution_for_cell(X_ref, x_query, n_pairs=1000, random_state=None):
    """
    Generate a null distribution of distances between a single query cell
    and random cells from the reference."""
    if random_state is not None:
        np.random.seed(random_state)
    n_ref = X_ref.shape[0]
    idx = np.random.choice(n_ref, size=n_pairs, replace=True)
    ref_samples = X_ref[idx]
    xq = x_query.reshape(-1) if x_query.ndim > 1 else x_query
    return np.linalg.norm(ref_samples - xq, axis=1)




def bootstrap_ci_for_cell(x_vals, weights, n_boot=500, alpha=0.05, random_state=None):
    """
    Compute bootstrap confidence interval for a weighted mean of values.
    - x_vals: (k,) pseudocluster values
    - weights: (k,) probabilities summing to 1
    Returns (lower, upper) quantiles."""
    if random_state is not None:
        np.random.seed(random_state)
    k = len(x_vals)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx_boot = np.random.choice(k, size=k, replace=True, p=weights)
        boot_means[b] = x_vals[idx_boot].mean()
    lower, upper = np.percentile(boot_means, [100 * (alpha/2), 100 * (1 - alpha/2)])
    return lower, upper



def impute_pseudocluster(adata_query, adata_ref, k=10,
                          n_null=1000, per_cell_null=False,
                          do_bootstrap=True, n_boot=500,
                          epsilon=1e-10):
    """
    Impute pseudocluster and compute confidence metrics.
    per_cell_null: if True, build a null per query cell;
                   if False, use group-level null.
    do_bootstrap:   if True, compute bootstrap CIs for each cell"""

    arc_path = '/home/shuonan.chen/scratch_shuonan/scripts/LC_NE_dataintegration/snRNAseq_only/cellID_pc_0722.csv'
    arcinfo = pd.read_csv(arc_path)
    assert (adata_ref.obs.index == arcinfo['cellID']).all()

    common = np.intersect1d(adata_query.var_names, adata_ref.var_names)
    print(f'using {len(common)} genes to run the imputations!')
    adata_ref = adata_ref[:, common].copy()

    # Extract and normalize
    X_ref = adata_ref[:, common].X
    X_q = adata_query[:, common].X
    X_ref_norm = processing.rankrows(X_ref, standardize=True)
    X_q_norm = processing.rankrows(X_q, standardize=True)

    # k-NN search
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_ref_norm)
    distances, indices = nbrs.kneighbors(X_q_norm)

    # Weights for imputation
    weights = 1.0 / (distances + epsilon)
    weights /= weights.sum(axis=1, keepdims=True)

    # Impute pseudocluster values
    imputed = np.zeros(adata_query.n_obs)
    for i in range(adata_query.n_obs):
        vals = arcinfo['pseudoclusters'].iloc[indices[i]].values
        imputed[i] = np.dot(weights[i], vals)

    # Mean NN distance per cell
    mean_nn = distances.mean(axis=1)

    # Null distribution and p-scores
    if per_cell_null:
        null_medians = np.zeros_like(mean_nn)
        pscores = np.zeros_like(mean_nn)
        for i in range(len(mean_nn)):
            d_rand = compute_null_distribution_for_cell(
                X_ref_norm, X_q_norm[i], n_pairs=n_null, random_state=42 + i
            )
            null_medians[i] = np.median(d_rand)
            pscores[i] = np.mean(d_rand > mean_nn[i])
    else:
        baseline_med, d_rand = compute_baseline_distance_between(
            X_ref_norm, X_q_norm, n_pairs=n_null, random_state=42
        )
        null_medians = np.full_like(mean_nn, baseline_med)
        pscores = (d_rand[:, None] > mean_nn[None, :]).mean(axis=0)

    # Bootstrap CIs (optional)
    ci_lower = None
    ci_upper = None
    if do_bootstrap:
        ci_lower = np.zeros(len(imputed))
        ci_upper = np.zeros(len(imputed))
        for i in range(len(imputed)):
            vals = arcinfo['pseudoclusters'].iloc[indices[i]].values
            ci_lower[i], ci_upper[i] = bootstrap_ci_for_cell(
                vals, weights[i], n_boot=n_boot, alpha=0.05, random_state=100 + i
            )
    
    conf_dict = {
        'conf_power': pscores,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
    return imputed, conf_dict

