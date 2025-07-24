# imputations 
import pandas as pd
import anndata
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import rankdata
from . import processing

def compute_baseline_distance_between(X1, X2, n_pairs=3000, random_state=0):
    np.random.seed(random_state)
    n1 = X1.shape[0]  # X_sc
    n2 = X2.shape[0]  # X_mer
    idx1 = np.random.choice(n1, n_pairs)
    idx2 = np.random.choice(n2, n_pairs)
    d_rand = np.linalg.norm(X1[idx1] - X2[idx2], axis=1)
    distances = np.linalg.norm(X1[idx1] - X2[idx2], axis=1) # distance amongst 1000 pairs. 
    return np.median(distances),d_rand


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

    # Randomly select holdoff genes from the observed genes, if requested.
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

    adata_mer_imputed.uns['holdoff_genes'] = holdoff_genes.tolist()
#     adata_mer_imputed.obsm['spatial'] = adata_mer.obsm['spatial']
    
    baseline_dist,d_rand = compute_baseline_distance_between(X_sc_norm, X_mer_norm, n_pairs=1000, random_state=42)
        
    mean_nn_distances = np.mean(distances,axis=1)  # for each N_mer,    
    confidence_scores = 1 - (mean_nn_distances / baseline_dist)
    pscores = (d_rand[:, None] > mean_nn_distances[None, :]).mean(axis=0)
    confidence_vals = np.sum(distances<baseline_dist, 1)  
    # alternatively count % of nbrs that are above baseline.
    frac_vals = confidence_vals / k
    
    adata_mer_imputed.obs['impute_quality'] = confidence_scores # score -oo to 1..
    adata_mer_imputed.obs['not_confident'] = confidence_scores<0 # binary 
    adata_mer_imputed.obs['confidence_vals'] = confidence_vals
    adata_mer_imputed.obs['frac_vals'] = frac_vals
    adata_mer_imputed.obs['p_scores'] = pscores   
    # --> “what fraction of random distances exceed my cell’s average neighbor distance?”

    
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

