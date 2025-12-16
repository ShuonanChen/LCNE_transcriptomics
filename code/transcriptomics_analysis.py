import numpy as np

import scanpy as sc


########################################################
################ LENGTH noramlizations ################
########################################################

def gene_lengths_from_gtf(gtf_path):
    import pandas as pd 
    import gzip
    records = []
    open_func = gzip.open if gtf_path.endswith(".gz") else open
    with open_func(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if fields[2] != "exon":
                continue
            chrom, source, feature, start, end, score, strand, frame, attr = fields
            attr_dict = {
                x.split(" ")[0]: x.split(" ")[1].replace('"', '')
                for x in attr.strip(";").split("; ")}
            gene_id = attr_dict.get("gene_name", None)
            if gene_id is None:
                continue
            length = int(end) - int(start) + 1
            records.append((gene_id, length))
    df = pd.DataFrame(records, columns=["gene", "exon_len"])
    gene_lengths = df.groupby("gene")["exon_len"].sum()
    return gene_lengths



def tpm_normalize(X, gene_lengths_bp):
    import scipy.sparse as sp
    gene_lengths_kb = gene_lengths_bp / 1e3
    if sp.issparse(X):
        X_len = X.multiply(1.0 / gene_lengths_kb)
        scale = X_len.sum(axis=1).A1
        scale[scale == 0] = 1e-10
        X_tpm = X_len.multiply(1e6 / scale[:, None])
        return X_tpm

    else:
        X_len = X / gene_lengths_kb
        scale = X_len.sum(axis=1, keepdims=True)
        scale[scale == 0] = 1e-10
        return X_len / scale * 1e6
    
    
########################################################
##################### DE analysis  #####################
########################################################    

###### scanpy first ######
def DE_scanpy(adata_retro_BN,
              global_LFC_scanpy,
             n_genes,key,groups, visualize = False): # input is BN'ed versions
    '''
    you just need a normalized adata and we will return the dictionary 
    '''
    mask = (adata_retro_BN.var_names.str.startswith("Gm") |
            adata_retro_BN.var_names.str.endswith("Rik"))# | adata_retro_BN.var_names.str.startswith("CN"))
    adata_de = adata_retro_BN[:, ~mask].copy()
    adata_de.uns = {}
    adata_de.raw = None
    adata_de._var_names_cached = None
    sc.tl.rank_genes_groups(
        adata_de, groupby='injection_site',
        use_raw = False,
        method="wilcoxon",layer = 'log(CPM)',
        key_added=key, #rankby_abs=True,
    #     n_genes=50 # this is just NOT to use all the genes here. 
    )
    
    sc.pl.rank_genes_groups(adata_de,key=key)
    if visualize:
        sc.pl.rank_genes_groups_violin(adata_de,groups = groups, n_genes=5,size = 3, key = key)
    
    scanpy_genes = {}    
    for group in adata_de.obs['injection_site'].cat.categories:
        # this will give G x 5 (G is predefined in the rank_genes_groups)
        df = sc.get.rank_genes_groups_df(adata_de, group=group,key=key) 
        df = df[df['logfoldchanges'] >= global_LFC_scanpy]
        df = df.sort_values('pvals_adj', ascending=True)
        top_genes = df['names'].tolist()[:n_genes]
        scanpy_genes[group] = top_genes

    for k,v in scanpy_genes.items():
        print(k,len(v), v)    
 
    return(scanpy_genes)


###### scvi ######
def DE_scvi(adata_retro, model, global_LFC_scvi, n_genes):
    groupby_col = 'injection_site'
    scvi_de_results = {}
    categories = adata_retro.obs[groupby_col].cat.categories
    for category in categories:
        if category != 'nan':  # Skip thatlmus
            cell_idx_A = adata_retro.obs[groupby_col] == category
            cell_idx_B = adata_retro.obs[groupby_col] != category
            de_result_full = model.differential_expression(idx1=cell_idx_A, idx2=cell_idx_B, mode='change')
            problematic_mask = (de_result_full.index.str.startswith("Gm") | de_result_full.index.str.endswith("Rik") )#|
    #             de_result_full.index.str.startswith("CN"))            
            de_result_filtered = de_result_full.loc[~problematic_mask]            
            scvi_de_results[category] = de_result_filtered

    top_genes_dict_scvi = {}    
    for group, df in scvi_de_results.items():        
        df_filt = df[df['lfc_mean'] >= global_LFC_scvi]
        df_sorted = df_filt.sort_values('bayes_factor', ascending=False)
        top_genes = df_sorted.index.tolist()[:n_genes]
        problematic = [g for g in top_genes if any(x in g for x in ['Gm', 'Rik'])]
        top_genes_dict_scvi[group] = top_genes        
    scvi_genes= top_genes_dict_scvi.copy()
    return(scvi_genes)
    

   