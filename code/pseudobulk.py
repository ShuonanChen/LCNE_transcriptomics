import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

MOUSE_COL = "external_donor_name"
TARGET_COL = "injection_site"

def make_pseudobulk(adata, target_col, mouse_col=MOUSE_COL, agg="sum", use_raw=False,
                    covariate_cols=()):
    """Aggregate single cells into per-(mouse, target) pseudobulk profiles.

    covariate_cols : extra obs columns (e.g. "gender") carried into pb_meta via
        .first(). They must be constant within each (mouse, target) group -- which
        is the case for mouse-level covariates like sex -- otherwise .first()
        would silently pick one value; we assert this.
    """
    adata0 = adata.raw.to_adata() if (use_raw and adata.raw is not None) else adata
    if use_raw and adata.raw is not None: adata0.obs = adata.obs.copy()
    bad = adata0.var_names.str.startswith("Gm") | adata0.var_names.str.endswith("Rik")
    ad = adata0[:, ~bad].copy()
    X = ad.X.toarray() if not isinstance(ad.X, np.ndarray) else ad.X
    expr = pd.DataFrame(X, index=ad.obs_names, columns=ad.var_names)
    obs = ad.obs.copy(); group_cols = [mouse_col, target_col]
    obs["__group_id"] = obs[group_cols].astype(str).agg("|".join, axis=1)
    gb = expr.groupby(obs["__group_id"])
    if agg == "mean": pb_expr = gb.mean()
    elif agg == "sum": pb_expr = gb.sum()
    else: raise ValueError(f"Unknown agg='{agg}', use 'sum' or 'mean'.")
    meta_cols = group_cols + [c for c in covariate_cols if c not in group_cols]
    for c in covariate_cols:
        n_levels = obs.groupby("__group_id")[c].nunique(dropna=False)
        if (n_levels > 1).any():
            bad_groups = n_levels.index[n_levels > 1].tolist()
            raise ValueError(
                f"Covariate '{c}' is not constant within {len(bad_groups)} group(s) "
                f"(e.g. {bad_groups[:3]}); it cannot be carried as a per-sample covariate.")
    pb_meta = obs.groupby("__group_id")[meta_cols].first()
    pb_expr.index = pb_meta.index
    return pb_expr, pb_meta



def counts_to_logCPM(pb_expr, cpm_scale=1e6, gene_lengths_bp=None):
    X = pb_expr.values.astype(float)
    if gene_lengths_bp is not None:
        gl = np.maximum(np.asarray(gene_lengths_bp, float) / 1000.0, 1e-6)
        X = X / gl[None, :]
    lib = np.maximum(X.sum(axis=1, keepdims=True), 1.0)
    return pd.DataFrame(np.log1p(X * (cpm_scale / lib)), index=pb_expr.index, columns=pb_expr.columns)


def de_pseudobulk_one_vs_all(pb_expr_log, pb_meta, target_col):
    targets = pb_meta[target_col].astype(str)
    unique_targets = sorted(targets.unique())
    genes = pb_expr_log.columns
    records = []
    for target in unique_targets:
        idx_target = targets == target
        idx_others = targets != target
        X_target = pb_expr_log.loc[idx_target]
        X_others = pb_expr_log.loc[idx_others]
        mean_target = X_target.mean(axis=0)
        mean_others = X_others.mean(axis=0)

        t_vals, p_vals = stats.ttest_ind(X_target.values, X_others.values, axis=0, equal_var=False, nan_policy="omit")
        log2fc = (mean_target - mean_others) / np.log(2)
        for g, lfc, m_t, m_o, tval, pval in zip(genes, log2fc, mean_target, mean_others, t_vals, p_vals):
            records.append({
                "gene": g, "target": target,
                "mean_target": m_t, "mean_others": m_o,
                "log2FC": lfc,"t": tval,"pval": pval,})
    de_df = pd.DataFrame.from_records(records)
    de_df["FDR"] = np.nan
    for target in de_df["target"].unique():  # multiple testing correction
        mask = de_df["target"] == target
        pvals = de_df.loc[mask, "pval"].values
        _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
        de_df.loc[mask, "FDR"] = qvals

    de_df = de_df.sort_values(["target", "FDR", "log2FC"],
                              ascending=[True, True, False])
    return de_df


def get_top_genes_pseudobulk(de_df,min_lfc,max_fdr=None,target_order=None):
    if target_order is None:
        target_order = sorted(de_df["target"].astype(str).unique())
    top_genes_dict = {}
    for target in target_order:
        df_target = de_df[de_df["target"] == target]
        if df_target.empty:
            top_genes_dict[target] = []
            continue
        mask = (df_target["log2FC"] >= min_lfc)
        if max_fdr is not None:
            mask &= (df_target["FDR"] <= max_fdr)

        df_filt = df_target[mask].sort_values("pval", ascending=True)
        top_genes_dict[target] = df_filt["gene"].tolist()
    return top_genes_dict


def plot_pseudobulk_de_heatmap(de_df,top_genes_dict,target_order=None,min_lfc=1.0,cmap="bwr"):
    if target_order is None:
        target_order = list(top_genes_dict.keys())
    unique_genes = list(dict.fromkeys(g for genes in top_genes_dict.values() for g in genes))
    if len(unique_genes) == 0:
        raise ValueError("No genes to plot (top_genes_dict is empty).")
    expr_matrix = np.zeros((len(target_order), len(unique_genes)))
    for i, target in enumerate(target_order):
        df_target = de_df[de_df["target"] == target]
        for j, gene in enumerate(unique_genes):
            gene_data = df_target[df_target["gene"] == gene]
            if len(gene_data) > 0:
                expr_matrix[i, j] = gene_data["log2FC"].iloc[0]
    vmax = np.abs(expr_matrix).max()
    vmax = vmax if vmax > 0 else 1.0
    fig, ax = plt.subplots(figsize=(0.5 * len(unique_genes) + 2, 0.4 * len(target_order) + 2))
    im = ax.imshow(
        expr_matrix,
        cmap=cmap,
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,)
    ax.set_xticks(range(len(unique_genes)))
    ax.set_xticklabels(unique_genes, rotation=90)
    ax.set_yticks(range(len(target_order)))
    ax.set_yticklabels(target_order)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Log2 Fold Change")
    ax.set_title(f"Pseudobulk DE (LFC ≥ {min_lfc})")
    plt.tight_layout()
    return fig, ax


def run_pseudobulk_pipeline(
    adata,
    groupby_col,               # e.g. "injection_site"
    mouse_col=MOUSE_COL,       # e.g. "external_donor_name"
    use_raw=True,
    agg="sum",
    gene_lengths_dict=None,    # dict: gene -> length_bp (optional)
    min_lfc=1.0,
    max_fdr=None,
    target_order=None,
    cpm_scale=1e6,
    do_plot=True,
):
    """
    Full pseudobulk pipeline:
      AnnData -> pseudobulk counts -> logCPM -> DE -> top genes -> heatmap.
    Returns a dict of all intermediate results.
    
    about the input: 
        if adata.X is the TMP counts alreay then you may want to use raw counts, why ? 
        because normalizing the count before aggregating across the cells is strange. 
        (we should be aggregating and then running the length normalization as in step 2 here)
    """
    # 1) pseudobulk counts
    pb_expr, pb_meta = make_pseudobulk(
        adata,
        target_col=groupby_col,
        mouse_col=mouse_col,
        agg=agg,
        use_raw=use_raw,
    )

    # 2) gene lengths aligned to pb_expr columns, if provided
    gene_lengths_bp = None
    if gene_lengths_dict is not None:
        genelength_bp = pb_expr.columns.to_series().map(gene_lengths_dict)
        # fill missing with median of observed
        genelength_bp = genelength_bp.astype(float)
        genelength_bp = genelength_bp.fillna(genelength_bp.median())
        gene_lengths_bp = genelength_bp.values

    # 3) normalization to logCPM
    pb_expr_logcpm = counts_to_logCPM(
        pb_expr,
        cpm_scale=cpm_scale,
        gene_lengths_bp=gene_lengths_bp,
    )

    # 4) DE
    de_df = de_pseudobulk_one_vs_all(
        pb_expr_logcpm,
        pb_meta,
        target_col=groupby_col,
    )

    # 5) top genes per target
    if target_order is None:
        target_order = sorted(pb_meta[groupby_col].astype(str).unique())

    top_genes_dict = get_top_genes_pseudobulk(
        de_df,
        min_lfc=min_lfc,
        max_fdr=max_fdr,
        target_order=target_order,
    )

    # 6) plot
    fig, ax = None, None
    if do_plot:
        fig, ax = plot_pseudobulk_de_heatmap(
            de_df,
            top_genes_dict,
            target_order=target_order,
            min_lfc=min_lfc,
        )

    return {
        "pb_expr": pb_expr,
        "pb_meta": pb_meta,
        "pb_expr_logcpm": pb_expr_logcpm,
        "de_df": de_df,
        "top_genes_dict": top_genes_dict,
        "fig": fig,
        "ax": ax,
    }











#############################################################################
#####################     DESeq2 on pseudobulk  #####################
#############################################################################    
def _sanitize_level(s: str) -> str: return re.sub(r"\W+", "_", str(s)).strip("_")

def make_binary_contrast_metadata(
    pb_meta: pd.DataFrame,
    groupby_col: str,
    target: str,
    other_label: str = "Other",):
    target_level = _sanitize_level(target)
    binary_col = f"{groupby_col}_{target_level}_vs_other"

    metadata_bin = pb_meta.copy()
    metadata_bin[binary_col] = pd.Series(
        np.where(metadata_bin[groupby_col].astype(str) == str(target), target_level, other_label),
        index=metadata_bin.index).astype("category")
    metadata_bin[binary_col] = metadata_bin[binary_col].cat.set_categories([other_label, target_level])
    return metadata_bin, binary_col, target_level


def run_deseq2_one_vs_rest(
    pb_expr: pd.DataFrame,
    pb_meta: pd.DataFrame,
    groupby_col: str,
    target: str,
    *,
    counts_round: bool = True,
    other_label: str = "Other",
    n_cpus: int = 4,
    refit_cooks: bool = True,
    quiet: bool = False,
):
    """
    Run pydeseq2 on pseudobulk counts for a single target vs all others.

    Important: pb_expr should be *counts* (not logCPM). Use results["pb_expr"].
    Returns:
      res_df:   results_df with 'target' column
      dds:      fitted DeseqDataSet
      stats:    DeseqStats object
      meta_bin: metadata used
      binary_col, target_level: contrast components
    """
    pb_counts = pb_expr.copy()
    if counts_round:
        pb_counts = pb_counts.round()
    pb_counts = pb_counts.astype(int)

    meta_bin, binary_col, target_level = make_binary_contrast_metadata(
        pb_meta, groupby_col=groupby_col, target=target, other_label=other_label)
    assert pb_counts.index.equals(meta_bin.index), "pb_expr and pb_meta index mismatch."
    dds = DeseqDataSet(
        counts=pb_counts,
        metadata=meta_bin,
        design_factors=[binary_col],
        refit_cooks=refit_cooks,
        n_cpus=n_cpus,
    )
    dds.deseq2()

    contrast = [binary_col, target_level, other_label]  #  list of 3 strings of the form ["variable", "tested_level", "control_level"]
    stat_res = DeseqStats(dds, contrast=contrast, n_cpus=n_cpus)
    stat_res.summary() if not quiet else None

    res_df = stat_res.results_df.copy()
    res_df["target"] = target
    return {
        "res_df": res_df,
        "dds": dds,
        "stat_res": stat_res,
        "metadata_bin": meta_bin,
        "binary_col": binary_col,
        "target_level": target_level,
        "contrast": contrast,
    }


def run_deseq2_multifactor_one_vs_rest(
    pb_expr: pd.DataFrame,
    pb_meta: pd.DataFrame,
    groupby_col: str,
    covariate_cols=("gender",),
    *,
    min_total_count: int = 10,
    counts_round: bool = True,
    n_cpus: int = 4,
    refit_cooks: bool = True,
    lfc_shrink: bool = False,
    quiet: bool = True,
):
    """Fit ONE DeseqDataSet over all levels of ``groupby_col`` and extract each
    level's effect as a one-vs-rest contrast (level vs the mean of the others).

    Design is ``~ <covariate_cols> + <groupby_col>``, so dispersion is estimated
    once across all samples (more stable at 3-8 replicates than re-pooling a
    heterogeneous "Other" group per target, as ``run_deseq2_one_vs_rest`` does)
    and nuisance covariates such as ``gender`` are adjusted out.

    Important details:
      * ``pb_expr`` must be *counts* (use results["pb_expr"] / make_pseudobulk sums).
      * Samples whose ``groupby_col`` is NaN/"nan" (e.g. thalamus set to NaN
        upstream) are dropped -- they must not leak into the reference.
      * Genes with summed count < ``min_total_count`` are dropped before fitting
        to stabilize dispersion (pydeseq2 recommendation).
      * pydeseq2 0.5.2's ``ref_level`` does not reliably control the dropped level,
        so the reference is auto-detected from the fitted LFC columns. One-vs-rest
        is reference-invariant anyway.

    LFC shrinkage caveat: ``DeseqStats.lfc_shrink`` shrinks a *named* design
    coefficient, not an arbitrary numeric contrast, so apeglm shrinkage cannot be
    applied to these one-vs-rest contrasts in 0.5.2. ``lfc_shrink=True`` therefore
    only warns and proceeds unshrunk. To get real shrinkage later, either rank by
    ``stat`` or recast the contrast of interest as a pairwise ``site-vs-ref``
    coefficient and call ``lfc_shrink(coeff="<groupby_col>[T.<site>]")``.

    Returns dict: ``res_df`` (long, all sites, gene index + 'target'),
    ``results_by_site`` (per-site results_df), ``dds``, ``reference_site``,
    ``genes_kept``, ``contrasts`` (site -> numpy vector).
    """
    import re
    import warnings
    import formulaic

    covariate_cols = list(covariate_cols)

    # --- counts -> int ---
    pb_counts = pb_expr.copy()
    if counts_round:
        pb_counts = pb_counts.round()
    pb_counts = pb_counts.astype(int)

    meta = pb_meta.copy()
    assert pb_counts.index.equals(meta.index), "pb_expr and pb_meta index mismatch."

    # --- drop NaN-site samples (fixes thalamus/NaN leak into the reference) ---
    site_str = meta[groupby_col].astype(str)
    keep_site = meta[groupby_col].notna() & (site_str.str.lower() != "nan")
    n_drop_site = int((~keep_site).sum())
    if n_drop_site:
        warnings.warn(f"Dropping {n_drop_site} pseudobulk sample(s) with NaN '{groupby_col}'.")
    meta = meta[keep_site]
    pb_counts = pb_counts.loc[meta.index]

    # --- drop covariate-NaN samples; require >=2 observed levels per covariate ---
    for c in covariate_cols:
        keep_cov = meta[c].notna() & (meta[c].astype(str).str.lower() != "nan")
        n_drop_cov = int((~keep_cov).sum())
        if n_drop_cov:
            warnings.warn(f"Dropping {n_drop_cov} pseudobulk sample(s) with NaN covariate '{c}'.")
            meta = meta[keep_cov]
            pb_counts = pb_counts.loc[meta.index]
        n_levels = meta[c].astype(str).nunique()
        if n_levels < 2:
            raise ValueError(
                f"Covariate '{c}' has {n_levels} level(s) after filtering; not estimable. "
                f"Drop it from covariate_cols.")

    # design factors must be categorical for pydeseq2
    for c in covariate_cols + [groupby_col]:
        meta[c] = meta[c].astype(str).astype("category")

    # --- low-count gene pre-filter ---
    genes_kept = pb_counts.columns[pb_counts.sum(axis=0) >= min_total_count]
    pb_counts = pb_counts[genes_kept]

    design_factors = covariate_cols + [groupby_col]
    design_str = "~ " + " + ".join(design_factors)

    # --- full-rank guard: a rank-deficient design makes dds.deseq2() HANG ---
    M = formulaic.model_matrix(design_str, meta)
    M = np.asarray(M)
    rank = np.linalg.matrix_rank(M)
    if rank < M.shape[1]:
        raise ValueError(
            f"Design '{design_str}' is rank-deficient ({rank} < {M.shape[1]} cols) -- "
            f"a covariate is confounded with '{groupby_col}'. Refusing to fit "
            f"(pydeseq2 would hang). Drop the confounded covariate.")

    dds = DeseqDataSet(
        counts=pb_counts,
        metadata=meta,
        design_factors=design_factors,
        refit_cooks=refit_cooks,
        n_cpus=n_cpus,
        quiet=quiet,
    )
    dds.deseq2()

    if lfc_shrink:
        warnings.warn(
            "lfc_shrink is not available for one-vs-rest numeric contrasts in pydeseq2 "
            "0.5.2; proceeding with unshrunk LFCs. See docstring for alternatives.")

    # --- map each site to its LFC coefficient column; reference = missing site ---
    lfc_cols = list(dds.varm["LFC"].columns)
    pat = re.compile(re.escape(groupby_col) + r"\[T\.(.+)\]")
    site_to_col = {}
    for j, col in enumerate(lfc_cols):
        m = pat.fullmatch(col)
        if m:
            site_to_col[m.group(1)] = j
    all_sites = sorted(meta[groupby_col].astype(str).unique())
    ref_sites = [s for s in all_sites if s not in site_to_col]
    assert len(ref_sites) == 1, f"Expected exactly one reference site, got {ref_sites}."
    reference_site = ref_sites[0]
    k = len(all_sites)

    def _ovr_contrast(target):
        vec = np.zeros(len(lfc_cols))
        for site, j in site_to_col.items():
            w = (1.0 if site == target else 0.0) - ((1.0 / (k - 1)) if site != target else 0.0)
            vec[j] = w
        return vec

    results_by_site = {}
    contrasts = {}
    frames = []
    for site in all_sites:
        vec = _ovr_contrast(site)
        contrasts[site] = vec
        stat_res = DeseqStats(dds, contrast=vec, n_cpus=n_cpus, quiet=quiet)
        stat_res.summary()
        rdf = stat_res.results_df.copy()
        rdf["target"] = site
        results_by_site[site] = rdf
        frames.append(rdf)

    res_df = pd.concat(frames, axis=0)
    return {
        "res_df": res_df,
        "results_by_site": results_by_site,
        "dds": dds,
        "reference_site": reference_site,
        "genes_kept": list(genes_kept),
        "contrasts": contrasts,
    }


def filter_deseq_enriched(
    res_df: pd.DataFrame,
    *,
    lfc_thr: float = 1.0,
    padj_thr: float = 0.05,
    lfc_col: str = "log2FoldChange",
    padj_col: str = "padj",
):
    df = res_df.copy()
    df = df[(df[lfc_col] > lfc_thr) & (df[padj_col] < padj_thr)]
    return df.sort_values(lfc_col, ascending=False)


def gene_set_overlap(genes_a, genes_b):
    set_a, set_b = set(genes_a), set(genes_b)
    return {
        "overlap": set_a & set_b,
        "only_a": set_a - set_b,
        "only_b": set_b - set_a,
    }


def compare_deseq_vs_ttest(
    deseq_res_df: pd.DataFrame,
    top_genes_dict_pseudobulk: dict,
    target: str,
    *,
    lfc_thr: float = 1.0,
    padj_thr: float = 0.05,
):
    """filter DESeq enriched genes and compare to t-test list."""
    enriched_df = filter_deseq_enriched(deseq_res_df, lfc_thr=lfc_thr, padj_thr=padj_thr)
    genes_deseq = enriched_df.index.astype(str).tolist()
    genes_ttest = list(top_genes_dict_pseudobulk.get(target, []))
    overlap = gene_set_overlap(genes_deseq, genes_ttest)
    return {
        "enriched_df": enriched_df,
        "genes_deseq": genes_deseq,
        "genes_ttest": genes_ttest,
        "overlap": overlap,
    }


