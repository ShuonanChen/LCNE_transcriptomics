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
    Dict-like: keys = group, values = list of top genes.
    Stores full DE tables and provides plotting utilities.
    """
    def __init__(
        self,
        genes_dict,
        de_results,
        groupby_col,
        global_LFC_scvi,
        n_genes,
        min_frac_expr=None,
        min_proba_de=None,
    ):
        super().__init__(genes_dict)
        self.de_results = de_results          # {group: full DE DataFrame}
        self.groupby_col = groupby_col
        self.global_LFC_scvi = global_LFC_scvi
        self.n_genes = n_genes
        self.min_frac_expr = min_frac_expr
        self.min_proba_de = min_proba_de

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

    # ------------------------------------------------------------------
    # 1) DE / LFC heatmap (sparse; NaNs where gene not in group's DE df)
    # ------------------------------------------------------------------
    def plotheatmap(
        self,
        groups=None,
        finalist_genes=None,
        value_col="lfc_mean",
        cmap="bwr",
        figsize=None,
        title=None,
        vlim=None,
    ):
        if groups is None:
            groups = self._default_groups()
        if finalist_genes is None:
            finalist_genes = self._default_finalist_genes(groups)

        if len(groups) == 0 or len(finalist_genes) == 0:
            raise ValueError("No groups or genes to plot in DE heatmap.")

        expr_matrix = np.full(
            (len(groups), len(finalist_genes)), np.nan, dtype=float
        )
        for i, group in enumerate(groups):
            df = self.de_results[group]
            for j, gene in enumerate(finalist_genes):
                if gene in df.index and value_col in df.columns:
                    expr_matrix[i, j] = float(df.loc[gene, value_col])

        if vlim is None:
            vmax = np.nanmax(np.abs(expr_matrix))
            vmin = -vmax
        else:
            vmin, vmax = vlim

        if figsize is None:
            figsize = (
                max(5, 0.25 * len(finalist_genes)),
                max(2, 0.35 * len(groups)),
            )

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            np.ma.masked_invalid(expr_matrix),
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xticks(range(len(finalist_genes)))
        ax.set_xticklabels(finalist_genes, rotation=90)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(value_col)

        if title is None:
            title = f"DE genes heatmap ({value_col}; LFC ≥ {self.global_LFC_scvi})"
        ax.set_title(title)
        plt.tight_layout()
        return ax

    # ------------------------------------------------------------------
    # 2) Expression heatmap (dense; mean expression per group & gene)
    # ------------------------------------------------------------------
    def plot_expression_heatmap(
        self,
        adata,
        layer=None,          # e.g. "log(CPM)" or "counts"; None -> use .X
        groups=None,
        genes=None,
        zscore=True,
        cmap="bwr",
        figsize=None,
        title=None,
    ):
        if groups is None:
            groups = self._default_groups()
        if genes is None:
            genes = self._default_finalist_genes(groups)

        # keep only groups that exist in adata
        groups = [
            g for g in groups
            if g in adata.obs[self.groupby_col].cat.categories
        ]

        if len(groups) == 0 or len(genes) == 0:
            raise ValueError("No groups or genes to plot in expression heatmap.")

        expr = np.zeros((len(groups), len(genes)), dtype=float)

        for i, g in enumerate(groups):
            idx = adata.obs[self.groupby_col] == g
            if layer is None:
                X = adata[idx, genes].X
            else:
                X = adata[idx, genes].layers[layer]
            m = X.mean(axis=0)
            m = m.A1 if hasattr(m, "A1") else m
            expr[i] = m

        if zscore:
            means = expr.mean(axis=0, keepdims=True)
            stds = expr.std(axis=0, keepdims=True)
            stds[stds == 0] = 1.0
            expr = (expr - means) / stds

        if figsize is None:
            figsize = (
                max(5, 0.25 * len(genes)),
                max(2, 0.35 * len(groups)),
            )

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(expr, aspect="auto", cmap=cmap)

        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(genes, rotation=90)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups)

        label = "z-scored mean expression" if zscore else "mean expression"
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label)

        if title is None:
            title = f"Expression heatmap (n={len(genes)} genes)"
        ax.set_title(title)

        plt.tight_layout()
        return ax

    # ------------------------------------------------------------------
    # 3) Confidence heatmap (dense; e.g. proba_de or bayes_factor)
    # ------------------------------------------------------------------
    def plot_confidence_heatmap(
        self,
        value_col="proba_de",
        threshold=0.9,          # values below this are white
        groups=None,
        genes=None,
        cmap="viridis",
        figsize=None,
        title=None,
        vlim=(0.0, 1.0),
    ):
        if groups is None:
            groups = self._default_groups()
        if genes is None:
            genes = self._default_finalist_genes(groups)

        if len(groups) == 0 or len(genes) == 0:
            raise ValueError("No groups or genes to plot in confidence heatmap.")

        conf = np.full((len(groups), len(genes)), np.nan, dtype=float)

        for i, g in enumerate(groups):
            df = self.de_results[g]
            for j, gene in enumerate(genes):
                if gene in df.index and value_col in df.columns:
                    conf[i, j] = float(df.loc[gene, value_col])

        # mask:
        # - NaNs
        # - values below threshold
        mask = np.isnan(conf) | (conf < threshold)
        conf_masked = np.ma.masked_array(conf, mask=mask)

        # colormap with white for masked values
        cmap_obj = plt.cm.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="white")

        if figsize is None:
            figsize = (
                max(5, 0.25 * len(genes)),
                max(2, 0.35 * len(groups)),
            )

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            conf_masked,
            aspect="auto",
            cmap=cmap_obj,
            vmin=vlim[0],
            vmax=vlim[1],
        )

        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(genes, rotation=90)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(value_col)

        if title is None:
            title = f"{value_col} (threshold ≥ {threshold})"
        ax.set_title(title)

        plt.tight_layout()
        return ax


# ----------------------------------------------------------------------
# DE_scvi helper
# ----------------------------------------------------------------------
def DE_scvi(
    adata_retro,
    model,
    global_LFC_scvi=1.0,   # effect size: log2 fold-change >= 1
    n_genes=10,            # max genes per group
    min_frac_expr=0.3,    # fraction of cells expressing in group A
    groupby_col="injection_site",
):
    min_proba_de = 0.5
    scvi_de_results = {}   # full DE tables per group (after Gm/Rik filter)
    top_genes_dict_scvi = {}

    categories = adata_retro.obs[groupby_col].cat.categories

    for category in categories:
        if category == "nan":
            continue

        cell_idx_A = adata_retro.obs[groupby_col] == category
        cell_idx_B = ~cell_idx_A

        de_result_full = model.differential_expression(
            idx1=cell_idx_A,
            idx2=cell_idx_B,
            mode="change",
        )

        # remove Gm / Rik genes
        problematic_mask = (
            de_result_full.index.str.startswith("Gm")
            | de_result_full.index.str.endswith("Rik")
        )
        df_full = de_result_full.loc[~problematic_mask].copy()

        # store full table
        scvi_de_results[category] = df_full

        # selection mask for "top genes"
        mask_sel = (
            (df_full["lfc_mean"] >= global_LFC_scvi)
            & (df_full["non_zeros_proportion1"] >= min_frac_expr)
            & (df_full["proba_de"] >= min_proba_de)
        )

        df_sel = df_full.loc[mask_sel].sort_values(
            "bayes_factor", ascending=False
        )
        top_genes_dict_scvi[category] = df_sel.index.tolist()[:n_genes]

    return ScviDEOutput(
        genes_dict=top_genes_dict_scvi,
        de_results=scvi_de_results,
        groupby_col=groupby_col,
        global_LFC_scvi=global_LFC_scvi,
        n_genes=n_genes,
        min_frac_expr=min_frac_expr,
        min_proba_de=min_proba_de,
    )