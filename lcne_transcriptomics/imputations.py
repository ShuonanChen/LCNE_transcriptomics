# imputations 
import pandas as pd
import anndata
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import rankdata
import processing

def compute_baseline_distance_between(X1, X2, n_pairs=3000, random_state=0):
    np.random.seed(random_state)
    n1 = X1.shape[0]  # X_sc
    n2 = X2.shape[0]  # X_mer
    idx1 = np.random.choice(n1, n_pairs)
    idx2 = np.random.choice(n2, n_pairs)
    d_rand = np.linalg.norm(X1[idx1] - X2[idx2], axis=1)
    distances = np.linalg.norm(X1[idx1] - X2[idx2], axis=1) # distance amongst 1000 pairs. 
    return np.median(distances),d_rand




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
    """note this is just for pseudoclusters
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





def impute_mer_data(adata_sc, adata_mer, k=10, n_hvg=1000, n_holdoff_genes = 0, random_state = 111):
    adata_sc = adata_sc.copy()  # work on a copy to avoid modifying the original data
    if n_hvg is not None:
        import scanpy as sc
        sc.pp.highly_variable_genes(adata_sc, n_top_genes=n_hvg, flavor='seurat_v3')
        hvg_genes = adata_sc.var_names[adata_sc.var['highly_variable']]
    else:
        hvg_genes = adata_sc.var_names

    union_genes = np.union1d(hvg_genes, adata_mer.var_names)  # 1183    
    print('union! ',len(union_genes))
    
    if n_holdoff_genes > 0:
        np.random.seed(random_state)
        holdoff_genes = np.random.choice(adata_mer.var_names, size=n_holdoff_genes, replace=False)
    else:
        holdoff_genes = np.array([])

    used_genes = np.setdiff1d(adata_mer.var_names, holdoff_genes)
    common_genes = np.intersect1d(used_genes, union_genes)
    adata_sc_union = adata_sc[:, union_genes].copy()  # we will impute based on this guy, 1183 genes in total     

    X_sc = adata_sc_union[:, common_genes].X
    X_mer = adata_mer[:, common_genes].X     
    X_sc_union = adata_sc_union.X 
    
    X_sc_norm = processing.rankrows(X_sc,standardize = True)
    X_mer_norm = processing.rankrows(X_mer,standardize = True)

    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_sc_norm)
    distances, indices = nbrs.kneighbors(X_mer_norm) # distance: size (N_mer, K)

    epsilon = 1e-10
    weights = 1 / (distances + epsilon)
    weights = weights / weights.sum(axis=1, keepdims=True)  # normalize so that weights sum to 1

    imputed_expr = np.zeros((adata_mer.n_obs, len(union_genes)))
    for i in range(adata_mer.n_obs):
        neighbor_idx = indices[i]
        if hasattr(X_sc_union, "todense"):
            imputed_expr[i, :] = np.dot(weights[i], X_sc_union[neighbor_idx, :].todense())
        else:
            imputed_expr[i, :] = np.dot(weights[i], X_sc_union[neighbor_idx, :])

    adata_mer_imputed = anndata.AnnData(
        X=imputed_expr,
        obs=adata_mer.obs.copy(),
        var=pd.DataFrame(index=union_genes))

    adata_mer_imputed.uns['holdoff_genes'] = holdoff_genes.tolist() # output the holdoffgenes 
#     adata_mer_imputed.obsm['spatial'] = adata_mer.obsm['spatial']
    
    baseline_dist,d_rand = compute_baseline_distance_between(X_sc_norm, X_mer_norm, n_pairs=1000, random_state=42)
        
    mean_nn_distances = np.mean(distances,axis=1)  # for each N_mer,    
    confidence_scores = 1 - (mean_nn_distances / baseline_dist)
    pscores = (d_rand[:, None] > mean_nn_distances[None, :]).mean(axis=0)
    confidence_vals = np.sum(distances<baseline_dist, 1)  
    # alternatively count % of nbrs that are above baseline.
    frac_vals = confidence_vals / k
    
    # adata_mer_imputed.obs['impute_quality'] = confidence_scores # score -oo to 1..
    # adata_mer_imputed.obs['not_confident'] = confidence_scores<0 # binary 
    # adata_mer_imputed.obs['confidence_vals'] = confidence_vals
    # adata_mer_imputed.obs['frac_vals'] = frac_vals
    
    adata_mer_imputed.obs['p_scores'] = pscores   # fraction of random distances which exceed the cells average neighbor distance
    return adata_mer_imputed
    
    
def replace_imputedvalues(adata_mer,adata_mer_imputed, genelist):
    '''
    (this should not be a necessary functions...)
    use adata_mer_imputed, pick genes from genelist, and add to adata_mer.obs as 'imput_genes'
    '''
    for g in genelist:
        adata_mer.obs[f'imput_{g}'] = (
            adata_mer_imputed[:, g].X.toarray().flatten() 
            if hasattr(adata_mer_imputed.X, "toarray") 
            else np.array(adata_mer_imputed[:, g].X.flatten()))
    return(adata_mer)



def impute_pseudocluster(adata_query, adata_ref, pc_dir,
                         k=10, n_null=1000, per_cell_null=False,
                         do_bootstrap=True, n_boot=500,
                         epsilon=1e-10, weighted=True):
    arc_path = pc_dir+'/cellID_pc_0722.csv'
    arcinfo = pd.read_csv(arc_path)
    assert (adata_ref.obs.index == arcinfo['cellID']).all()

    common = np.intersect1d(adata_query.var_names, adata_ref.var_names)
    print(f'using {len(common)} genes to run the imputations!')
    adata_ref = adata_ref[:, common].copy()

    X_ref = adata_ref[:, common].X
    X_q = adata_query[:, common].X
    X_ref_norm = processing.rankrows(X_ref, standardize=True)
    X_q_norm = processing.rankrows(X_q, standardize=True)

    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_ref_norm)
    distances, indices = nbrs.kneighbors(X_q_norm)

    weights = 1.0 / (distances + epsilon)
    weights /= weights.sum(axis=1, keepdims=True)

    imputed = np.zeros(adata_query.n_obs)
    pc_std = np.zeros(adata_query.n_obs)
    for i in range(adata_query.n_obs):
        vals = arcinfo['pseudoclusters'].iloc[indices[i]].values.astype(float)
        w = weights[i] if weighted else np.full_like(weights[i], 1.0 / len(weights[i]))
        mu = np.dot(w, vals)
        imputed[i] = mu
        pc_std[i] = np.sqrt(np.sum(w * (vals - mu) ** 2))

    mean_nn = distances.mean(axis=1)

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

    ci_lower = None
    ci_upper = None
    if do_bootstrap:
        ci_lower = np.zeros(len(imputed))
        ci_upper = np.zeros(len(imputed))
        for i in range(len(imputed)):
            vals = arcinfo['pseudoclusters'].iloc[indices[i]].values.astype(float)
            w = weights[i] if weighted else np.full_like(weights[i], 1.0 / len(weights[i]))
            ci_lower[i], ci_upper[i] = bootstrap_ci_for_cell(
                vals, w, n_boot=n_boot, alpha=0.05, random_state=100 + i
            )

    conf_dict = {
        'conf_power': pscores,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'pc_std': pc_std
    }
    return imputed, conf_dict    

