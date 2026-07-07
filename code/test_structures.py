import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Continuum-vs-discrete testing (dip test + dip-dist)
#
# Hartigan's dip test is intrinsically 1-D. To ask "continuum vs discrete" in a
# multivariate embedding we use dip-dist (Dip-means, Kalogeratos & Likas 2012):
# for each "viewer" cell, dip-test the distribution of its distances to all other
# cells. One connected structure (a blob OR an arc) -> unimodal distances;
# >=2 separated clumps -> multimodal distances. See dip_dist_multivariate().
#
# Backend: uses the `diptest` package if installed (fast, calibrated Hartigan
# p-values); otherwise falls back to a pure-scipy Silverman critical-bandwidth
# unimodality test so everything runs before the env is rebuilt.
# ---------------------------------------------------------------------------

def _dip_backend():
    """Return 'diptest' if the package is importable, else 'silverman'."""
    try:
        import diptest  # noqa: F401
        return 'diptest'
    except ImportError:
        return 'silverman'


def _count_kde_modes(x, h, grid_size=256):
    """Number of local maxima of a Gaussian KDE of `x` with kernel std `h`."""
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    if sd == 0 or h <= 0:
        return 1
    kde = gaussian_kde(x, bw_method=h / sd)  # gaussian_kde kernel std = factor*sd
    grid = np.linspace(x.min() - 3 * h, x.max() + 3 * h, grid_size)
    dens = kde(grid)
    # count + -> - transitions of the first difference (interior local maxima)
    dd = np.diff(dens)
    return int(np.sum((dd[:-1] > 0) & (dd[1:] <= 0)))


def _critical_bandwidth(x, n_modes=1, max_iter=22):
    """Smallest kernel std h for which the KDE of `x` has <= n_modes modes.

    Mode count is monotone non-increasing in h for a Gaussian kernel, so we
    binary-search h in [0, data range]. max_iter fixes cost (~2^-22 precision).
    """
    x = np.asarray(x, float)
    rng = x.max() - x.min()
    if rng == 0:
        return 0.0
    lo, hi = 0.0, rng  # a single wide kernel (h=range) over-smooths to 1 mode
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _count_kde_modes(x, mid) <= n_modes:
            hi = mid
        else:
            lo = mid
    return hi


def _silverman_pvalue(x, n_boot=500, random_state=0):
    """Silverman (1981) critical-bandwidth test of unimodality.

    Null: density is unimodal. Statistic: critical bandwidth h_crit. p-value via
    smoothed bootstrap from the h_crit KDE: fraction of bootstrap samples whose
    own critical bandwidth >= h_crit. Small p => reject unimodality (multimodal).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    sd = x.std(ddof=1)
    if n < 8 or sd == 0:
        return 0.0, 1.0
    h_crit = _critical_bandwidth(x, n_modes=1)
    if h_crit == 0:
        return 0.0, 1.0
    rng = np.random.default_rng(random_state)
    mean = x.mean()
    var = sd ** 2
    shrink = 1.0 / np.sqrt(1.0 + h_crit ** 2 / var)  # variance-preserving rescale
    count = 0
    for _ in range(n_boot):
        base = rng.choice(x, size=n, replace=True)
        smooth = base + h_crit * rng.standard_normal(n)
        y = mean + shrink * (smooth - mean)
        if _critical_bandwidth(y, n_modes=1) >= h_crit:
            count += 1
    return h_crit, count / n_boot


def _dip_pvalue(x, n_boot=500, random_state=0):
    """Unimodality test of a 1-D sample. Returns (statistic, pvalue, backend).

    Uses Hartigan's dip (`diptest`) when available, else the Silverman fallback.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return 0.0, 1.0, 'na'
    if _dip_backend() == 'diptest':
        import diptest
        dip, pval = diptest.diptest(x)
        return float(dip), float(pval), 'diptest'
    stat, pval = _silverman_pvalue(x, n_boot=n_boot, random_state=random_state)
    return float(stat), float(pval), 'silverman'


def unimodality_test_1d(x, n_boot=500, random_state=0):
    """Test whether a 1-D sample (e.g. the pseudocluster coordinate) is unimodal.

    NOTE: on `scores_01` this is a *secondary, conditional* readout -- the
    coordinate is built by fitting the arc, so unimodality is uninformative;
    only multimodality (peaks) is evidence of discrete stops along the arc.
    """
    stat, pval, backend = _dip_pvalue(x, n_boot=n_boot, random_state=random_state)
    xx = np.asarray(x, float)
    xx = xx[np.isfinite(xx)]
    # descriptive mode count at Scott's-rule bandwidth (not the critical bandwidth,
    # which is unimodal by construction)
    if xx.size >= 8 and xx.std(ddof=1) > 0:
        h_scott = xx.std(ddof=1) * xx.size ** (-1.0 / 5.0)
        n_modes_est = _count_kde_modes(xx, h_scott)
    else:
        n_modes_est = 1
    return {'stat': stat, 'pval': pval, 'backend': backend, 'n_modes_est': n_modes_est}


def _bh_reject(pvals, alpha=0.05):
    """Benjamini-Hochberg: boolean reject mask at FDR=alpha."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    reject = np.zeros(n, bool)
    if passed.any():
        kmax = np.max(np.where(passed))
        reject[order[:kmax + 1]] = True
    return reject


def dip_dist_multivariate(Z, n_viewers=200, n_ref=None, alpha=0.05,
                          n_boot=None, random_state=0, plot=True, label=''):
    """dip-dist (Dip-means) continuum-vs-discrete test on a multivariate embedding.

    For each viewer cell, dip-test the distribution of its Euclidean distances to
    other cells. `frac_split` = fraction of viewers whose distance distribution is
    significantly multimodal (BH-corrected) = discreteness score (high => discrete).

    Parameters
    ----------
    Z : (n_cells, n_dims) array (e.g. Ztr_pca)
    n_viewers : random subset of cells to use as viewers (None/>=n => all)
    n_ref     : random subset of other cells for the distance distribution
                (None => all); subsample for speed on large data
    n_boot    : Silverman bootstrap reps (ignored by diptest); default 150 for
                the silverman backend, else unused
    """
    Z = np.asarray(Z, float)
    n = Z.shape[0]
    rng = np.random.default_rng(random_state)
    backend = _dip_backend()
    if n_boot is None:
        n_boot = 150 if backend == 'silverman' else 0

    viewers = (np.arange(n) if (n_viewers is None or n_viewers >= n)
               else rng.choice(n, size=n_viewers, replace=False))

    pvals, stats, dists_keep = [], [], []
    for vi in viewers:
        d = np.linalg.norm(Z - Z[vi], axis=1)
        d = np.delete(d, vi)
        if n_ref is not None and n_ref < d.size:
            d = rng.choice(d, size=n_ref, replace=False)
        s, p, backend = _dip_pvalue(d, n_boot=n_boot, random_state=int(rng.integers(1e9)))
        pvals.append(p); stats.append(s); dists_keep.append(d)

    pvals = np.asarray(pvals); stats = np.asarray(stats)
    reject = _bh_reject(pvals, alpha=alpha)
    frac_split = float(reject.mean())

    out = {'frac_split': frac_split, 'pvals': pvals, 'dip_stats': stats,
           'reject': reject, 'backend': backend, 'n_viewers': len(viewers)}

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        # example distance histograms: most/least multimodal viewers
        order = np.argsort(pvals)
        for ax, idx, tag in [(axes[0], order[0], 'most multimodal'),
                             (axes[1], order[-1], 'least multimodal')]:
            ax.hist(dists_keep[idx], bins=40, color='0.4')
            ax.set_title(f'{tag}\np={pvals[idx]:.3g}')
            ax.set_xlabel('distance to other cells'); ax.set_ylabel('# cells')
        axes[2].hist(pvals, bins=20, range=(0, 1), color='steelblue')
        axes[2].axvline(alpha, color='red', ls='--')
        axes[2].set_title(f'{label}\nfrac_split={frac_split:.2f} ({backend})')
        axes[2].set_xlabel('viewer dip p-value'); axes[2].set_ylabel('# viewers')
        fig.tight_layout()
        out['fig'], out['axes'] = fig, axes

    return out



def compare_permuted_pca_cumulative(
    adata,
    n_perm_genes,
    n_pcs=50,
    n_perms=10,
    n_examples=3,
    seed=0,
    use_highly_variable=False,
    cmap_original='tab10',
    cmap_perm='Set2'
):
    assert use_highly_variable==False, "not accepting using HVGs for now, set it to false!"
    rng = np.random.default_rng(seed)

    # Real PCA
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack')
    orig_vr  = adata.uns['pca']['variance_ratio']
    orig_cum = np.cumsum(orig_vr)
    orig_pca = adata.obsm['X_pca'][:, :2]

    # Permutations
    perm_cums = []
    example_pcas = []
    for i in range(n_perms):
        ad = adata.copy()
        X = ad.X.toarray() if sp.issparse(ad.X) else ad.X.copy()
        cols = rng.choice(X.shape[1], size=n_perm_genes, replace=False)
        for gi in cols:
            rng.shuffle(X[:, gi])
        ad.X = X

        sc.tl.pca(ad, n_comps=n_pcs, svd_solver='arpack')
        perm_vr = ad.uns['pca']['variance_ratio']
        perm_cums.append(np.cumsum(perm_vr))
        if i < n_examples:
            example_pcas.append(ad.obsm['X_pca'][:, :2])

    perm_arr   = np.vstack(perm_cums)
    mean_perm  = perm_arr.mean(axis=0)
    std_perm   = perm_arr.std(axis=0)
    pcs        = np.arange(1, n_pcs+1)

    # Combined figure: left (cumulative), right (2x2 scatters)
    fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 6),
    gridspec_kw={'width_ratios': [1.4, 1]},
    constrained_layout=True)

    # Left cumulative variance plot
    ax_cum = axes[0]
    ax_cum.fill_between(pcs, mean_perm - std_perm, mean_perm + std_perm,
                        alpha=0.3, label=f'Permuted ±1 std\n({n_perm_genes} genes)', zorder=1)
    ax_cum.plot(pcs, mean_perm, marker='o', linestyle='--', label='Permuted mean', zorder=3)
    ax_cum.plot(pcs, orig_cum, marker='o', label='Original', zorder=4)
    ax_cum.axhline(orig_cum[10], linestyle=':', color='gray',
                   label=f'{np.round(orig_cum[10]*100)}% thresh', zorder=2)
    ax_cum.set(xlabel='Number of PCs',
               ylabel='Cumulative explained variance',
               title=f'Permute {n_perm_genes} genes — {n_perms} runs')
    ax_cum.legend(frameon=False)

    # Right 2x2 grid for PCA scatters
    axes[1].remove()
    gs = axes[1].get_subplotspec().subgridspec(2, 2)
    sub_axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    sub_axes[0].scatter(orig_pca[:,0], orig_pca[:,1],
                        s=5, alpha=0.7, c='pink', label='Original')
    sub_axes[0].set(title='Original PCA', xlabel='PC1', ylabel='PC2', aspect='equal')

    for j, perm_pca in enumerate(example_pcas[:3], start=1):
        sub_axes[j].scatter(perm_pca[:,0], perm_pca[:,1],
                            s=5, alpha=0.7, c='gray', label=f'Perm #{j}')
        sub_axes[j].set(title=f'Permuted PCA #{j}', xlabel='PC1', ylabel='PC2', aspect='equal')
    return fig, (ax_cum, sub_axes)


def plot_pca_hvg_variation(
    adata,  
    cpm_scl,
    hvg_counts=[500, 1000, 2000, 5000],
    flavor='seurat_v3',
    pca_solver='arpack',
    scatter_kwargs=None
):
    """
    For each n in hvg_counts:
      1) select top-n HVGs
      2) run PCA (2 components)
      3) scatter PC1 vs PC2
    
    Parameters
    ----------
    adata : AnnData
        Your full dataset. this should be before you running z-score normalization. in our case its adata_BN (sample sum is 1)
    hvg_counts : list of int
        Numbers of HVGs to try.
    flavor : str
        Scanpy HVG method.
    pca_solver : str
        SVD solver for sc.tl.pca.
    scatter_kwargs : dict, optional
        Passed to plt.scatter (e.g. {'s':5,'alpha':0.7,'c':adata.obs['cell_type']}).
    """
    if scatter_kwargs is None:
        scatter_kwargs = dict(s=5, alpha=0.7)
    n_plots = len(hvg_counts)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5), squeeze=False)


    adata_cpm = adata.copy()
    adata_cpm.X*=cpm_scl
    adata_cpm.X = np.around(adata_cpm.X)

    for ax, n_hvg in zip(axes[0], hvg_counts):
        # 1) copy and select HVGs
        ad = adata_cpm.copy()
        sc.pp.highly_variable_genes(ad, 
                                    n_top_genes=n_hvg, 
                                    flavor=flavor,
                                    subset=True)  # keeps only HVGs
        
        # so now you have to normalize!
        sc.pp.scale(ad, zero_center=True, max_value=10)
        # 2) compute PCA (2 PCs)        
        sc.tl.pca(ad, 
                  n_comps=2, 
                  svd_solver=pca_solver)
        
        # 3) scatter PC1 vs PC2
        pcs = ad.obsm['X_pca']  # shape (cells,2)
        ax.scatter(pcs[:,0], pcs[:,1], **scatter_kwargs)
        ax.set_title(f'{n_hvg} HVGs')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_aspect('equal')
    
    fig.tight_layout()
    


def plot_pca_hvg_variation(
    adata,  
    cpm_scl,
    hvg_counts=[500, 1000, 2000, 5000],
    flavor='seurat_v3',
    pca_solver='arpack',
    scatter_kwargs=None
):
    """
    For each n in hvg_counts:
      1) select top-n HVGs
      2) run PCA (2 components)
      3) scatter PC1 vs PC2
    
    Parameters
    ----------
    adata : AnnData
        Your full dataset. this should be before you running z-score normalization. in our case its adata_BN (sample sum is 1)
    hvg_counts : list of int
        Numbers of HVGs to try.
    flavor : str
        Scanpy HVG method.
    pca_solver : str
        SVD solver for sc.tl.pca.
    scatter_kwargs : dict, optional
        Passed to plt.scatter (e.g. {'s':5,'alpha':0.7,'c':adata.obs['cell_type']}).
    """
    if scatter_kwargs is None:
        scatter_kwargs = dict(s=5, alpha=0.7)
    n_plots = len(hvg_counts)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5), squeeze=False)

    adata_cpm = adata.copy()
    adata_cpm.X*=cpm_scl
    adata_cpm.X = np.around(adata_cpm.X)

    for ax, n_hvg in zip(axes[0], hvg_counts):
        # 1) copy and select HVGs
        ad = adata_cpm.copy()
        sc.pp.highly_variable_genes(ad, 
                                    n_top_genes=n_hvg, 
                                    flavor=flavor,
                                    subset=True)  # keeps only HVGs
        
        # so now you have to normalize!
        sc.pp.scale(ad, zero_center=True, max_value=10)
        # 2) compute PCA (2 PCs)        
        sc.tl.pca(ad, 
                  n_comps=2, 
                  svd_solver=pca_solver)
        
        # 3) scatter PC1 vs PC2
        pcs = ad.obsm['X_pca']  # shape (cells,2)
        ax.scatter(pcs[:,0], pcs[:,1], **scatter_kwargs)
        ax.set_title(f'{n_hvg} HVGs')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_aspect('equal')
    
    fig.tight_layout()
    return fig, axes
