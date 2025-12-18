import numpy as np
import matplotlib.pyplot as plt
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
class ScviDEOutput(dict):
    """
    Dict-like: keys = group, values = list of top genes
    Also stores full DE tables and can plot a heatmap of lfc_mean.
    """
    def __init__(self, genes_dict, de_results, groupby_col, global_LFC_scvi, n_genes):
        super().__init__(genes_dict)
        self.de_results = de_results
        self.groupby_col = groupby_col
        self.global_LFC_scvi = global_LFC_scvi
        self.n_genes = n_genes

    def _default_groups(self):
        return list(self.de_results.keys())

    def _default_finalist_genes(self, groups):
        seen, out = set(), []
        for g in groups:
            for gene in self.get(g, []):
                if gene not in seen:
                    seen.add(gene)
                    out.append(gene)
        return out

    def plotheatmap(self, groups=None, finalist_genes=None, value_col="lfc_mean",
                    cmap="bwr", figsize=None, title=None, vlim=None):
        if groups is None:
            groups = self._default_groups()
        if finalist_genes is None:
            finalist_genes = self._default_finalist_genes(groups)

        expr_matrix = np.zeros((len(groups), len(finalist_genes)), dtype=float)
        for i, group in enumerate(groups):
            df = self.de_results[group]
            for j, gene in enumerate(finalist_genes):
                if gene in df.index:
                    expr_matrix[i, j] = float(df.loc[gene, value_col])

        if vlim is None:
            vmax = np.abs(expr_matrix).max()
            vmin = -vmax
        else:
            vmin, vmax = vlim

        if figsize is None:
            figsize = (max(5, 0.25 * len(finalist_genes)), max(2, 0.35 * len(groups)))

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(expr_matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(finalist_genes)))
        ax.set_xticklabels(finalist_genes, rotation=90)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(value_col)

        if title is None:
            title = f"DE genes heatmap ({value_col}; LFC ≥ {self.global_LFC_scvi})"
        ax.set_title(title)

#         plt.tight_layout()
#         plt.show()
        return ax


def DE_scvi(adata_retro, model, global_LFC_scvi, n_genes):
    groupby_col = "injection_site"
    scvi_de_results = {}
    categories = adata_retro.obs[groupby_col].cat.categories

    for category in categories:
        if category != "nan":
            cell_idx_A = adata_retro.obs[groupby_col] == category
            cell_idx_B = adata_retro.obs[groupby_col] != category
            de_result_full = model.differential_expression(idx1=cell_idx_A, idx2=cell_idx_B, mode="change")
            problematic_mask = (de_result_full.index.str.startswith("Gm") | de_result_full.index.str.endswith("Rik"))
            de_result_filtered = de_result_full.loc[~problematic_mask]
            scvi_de_results[category] = de_result_filtered

    top_genes_dict_scvi = {}
    for group, df in scvi_de_results.items():
        df_filt = df[df["lfc_mean"] >= global_LFC_scvi]
        df_sorted = df_filt.sort_values("bayes_factor", ascending=False)
        top_genes_dict_scvi[group] = df_sorted.index.tolist()[:n_genes]

    return ScviDEOutput(
        genes_dict=top_genes_dict_scvi,
        de_results=scvi_de_results,
        groupby_col=groupby_col,
        global_LFC_scvi=global_LFC_scvi,
        n_genes=n_genes,
    )