"""
Linear mixed-effects models for the retroseq projection-target analysis.

Each mouse (``external_donor_name``) was injected at a single projection target
(``injection_site``), so mouse is *nested within site*. Because every mouse is
unique to one site, a random intercept grouped by mouse -- ``groups=mouse`` in
``statsmodels`` -- already encodes the site-nested random effect; no explicit
nesting syntax is needed.

The model fit per gene (one-vs-rest, to mirror ``pseudobulk.de_pseudobulk_one_vs_all``):

    log1pCPM_gene ~ C(site_vs_rest) + (1 | mouse)

The fixed site effect is then evaluated against *between-mouse* variance rather
than the (much larger) cell count, so significance is not inflated by treating
correlated cells from the same mouse as independent replicates.

Output ``de_lmm_df`` shares the key columns (``gene``, ``target``, ``log2FC``,
``pval``, ``FDR``) used by ``pseudobulk.get_top_genes_pseudobulk`` /
``pseudobulk.plot_pseudobulk_de_heatmap`` so it drops into the existing plots,
plus the batch-diagnostic columns ``mouse_var`` / ``resid_var`` / ``icc``.
"""
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import pseudobulk
from pseudobulk import MOUSE_COL, TARGET_COL, counts_to_logCPM, make_binary_contrast_metadata


def prep_lmm_matrix(
    adata,
    target_col=TARGET_COL,
    mouse_col=MOUSE_COL,
    drop=("thalamus",),
    n_top_genes=2000,
    cpm_scale=1e6,
    use_raw=False,
):
    """Subset to the projection sites of interest, log1p-CPM normalize, and pick HVGs.

    Returns
    -------
    expr_df : DataFrame (cells x HVGs) of log1p-CPM expression.
    meta    : DataFrame (cells x [target_col, mouse_col]) aligned to expr_df.
    """
    adata0 = adata.raw.to_adata() if (use_raw and adata.raw is not None) else adata
    if use_raw and adata.raw is not None:
        adata0.obs = adata.obs.copy()

    sites = adata0.obs[target_col].astype(str)
    drop_lower = {str(d).lower() for d in drop}
    keep = sites.notna() & ~sites.str.lower().isin(drop_lower) & (sites.str.lower() != "nan")
    ad = adata0[keep.values].copy()

    # drop predicted/uninformative gene models, matching make_pseudobulk
    bad = ad.var_names.str.startswith("Gm") | ad.var_names.str.endswith("Rik")
    ad = ad[:, ~bad].copy()

    X = ad.X.toarray() if not isinstance(ad.X, np.ndarray) else np.asarray(ad.X)
    counts = pd.DataFrame(X, index=ad.obs_names, columns=ad.var_names)
    logcpm = counts_to_logCPM(counts, cpm_scale=cpm_scale)

    # HVG selection on the log-normalized matrix
    hv = sc.AnnData(logcpm.values.copy())
    hv.var_names = logcpm.columns
    sc.pp.highly_variable_genes(hv, n_top_genes=min(n_top_genes, hv.shape[1]), flavor="seurat")
    hvgs = hv.var_names[hv.var["highly_variable"].values].tolist()

    expr_df = logcpm[hvgs]
    meta = ad.obs[[target_col, mouse_col]].copy()
    meta[target_col] = meta[target_col].astype(str)
    meta[mouse_col] = meta[mouse_col].astype(str)
    return expr_df, meta


def fit_lmm_gene(y, df, grp_col, mouse_col, target_level):
    """Fit ``y ~ C(grp_col) + (1 | mouse)`` for one gene.

    ``df`` must contain ``grp_col`` (binary site-vs-other, reference = other) and
    ``mouse_col``. Returns a dict; on non-convergence/singular fit the effect
    fields are NaN and ``converged`` is False.
    """
    out = {"coef": np.nan, "pval": np.nan, "mouse_var": np.nan,
           "resid_var": np.nan, "icc": np.nan, "converged": False}
    d = df[[grp_col, mouse_col]].copy()
    d["__expr"] = np.asarray(y, dtype=float)
    coef_name = f"C({grp_col})[T.{target_level}]"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm(f"__expr ~ C({grp_col})", d, groups=d[mouse_col])
            mdf = md.fit(reml=True)
        mouse_var = float(mdf.cov_re.iloc[0, 0])
        resid_var = float(mdf.scale)
        denom = mouse_var + resid_var
        out.update(
            coef=float(mdf.params.get(coef_name, np.nan)),
            pval=float(mdf.pvalues.get(coef_name, np.nan)),
            mouse_var=mouse_var,
            resid_var=resid_var,
            icc=(mouse_var / denom) if denom > 0 else np.nan,
            converged=bool(getattr(mdf, "converged", True)),
        )
    except Exception:
        pass
    return out


def run_lmm_de(expr_df, meta, target_col=TARGET_COL, mouse_col=MOUSE_COL):
    """One-vs-rest LMM differential expression across all HVGs and sites.

    Returns ``de_lmm_df`` with columns
    ``[gene, target, coef, log2FC, pval, FDR, mouse_var, resid_var, icc, converged]``.
    ``log2FC`` (= coef / ln2) and ``FDR`` (BH per target) match the pseudobulk
    output so the existing top-gene/heatmap helpers can be reused.
    """
    sites = sorted(meta[target_col].astype(str).unique())
    genes = list(expr_df.columns)
    records = []
    for target in sites:
        meta_bin, grp_col, target_level = make_binary_contrast_metadata(
            meta, groupby_col=target_col, target=target)
        df = meta_bin[[grp_col, mouse_col]]
        for g in genes:
            res = fit_lmm_gene(expr_df[g].values, df, grp_col, mouse_col, target_level)
            records.append({
                "gene": g, "target": target,
                "coef": res["coef"], "log2FC": res["coef"] / np.log(2),
                "pval": res["pval"], "mouse_var": res["mouse_var"],
                "resid_var": res["resid_var"], "icc": res["icc"],
                "converged": res["converged"],
            })
    de_df = pd.DataFrame.from_records(records)
    de_df["FDR"] = np.nan
    for target in de_df["target"].unique():
        mask = (de_df["target"] == target) & de_df["pval"].notna()
        if mask.sum() == 0:
            continue
        _, qvals, _, _ = multipletests(de_df.loc[mask, "pval"].values, method="fdr_bh")
        de_df.loc[mask, "FDR"] = qvals
    return de_df.sort_values(["target", "FDR", "log2FC"], ascending=[True, True, False])


def variance_partition_pcs(comp_matrix, meta, target_col=TARGET_COL, mouse_col=MOUSE_COL, n_comp=10):
    """Partition variance of each embedding component into site / mouse / residual.

    ``comp_matrix`` is a (cells x components) array aligned to ``meta`` (e.g.
    ``adata.obsm['X_pca']`` or the SCVI latent). For each component fits
    ``comp ~ C(site) + (1 | mouse)`` and reports the fraction of variance from the
    fixed site effect, the mouse (batch) random intercept, and the residual.
    """
    comp_matrix = np.asarray(comp_matrix)
    n_comp = min(n_comp, comp_matrix.shape[1])
    records = []
    for k in range(n_comp):
        d = meta[[target_col, mouse_col]].copy()
        d["__y"] = comp_matrix[:, k].astype(float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md = smf.mixedlm(f"__y ~ C({target_col})", d, groups=d[mouse_col])
                mdf = md.fit(reml=True)
            var_site = float(np.var(mdf.predict(d), ddof=0))   # fixed-effect (site) component
            var_mouse = float(mdf.cov_re.iloc[0, 0])
            var_resid = float(mdf.scale)
        except Exception:
            var_site = var_mouse = var_resid = np.nan
        total = var_site + var_mouse + var_resid
        records.append({
            "component": k,
            "var_site": var_site, "var_mouse": var_mouse, "var_resid": var_resid,
            "site_frac": var_site / total if total > 0 else np.nan,
            "mouse_frac": var_mouse / total if total > 0 else np.nan,
            "resid_frac": var_resid / total if total > 0 else np.nan,
        })
    return pd.DataFrame.from_records(records).set_index("component")
