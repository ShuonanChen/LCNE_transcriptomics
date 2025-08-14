import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt


def compare_permuted_pca_cumulative(
    adata,
    n_perm_genes,
    n_pcs=50,
    n_perms=10,
    n_examples=2,
    seed=0,
    use_highly_variable=False,
    cmap_original='tab10',
    cmap_perm='Set2'
):
    assert use_highly_variable==False, "not accepting using HVGs for now, set it to false!"
    
    """
    1) Plot original vs permuted cumulative explained variance.
    2) Show PC1–PC2 scatter of the real data plus `n_examples` permuted runs.

    Returns
    -------
    fig_cum, ax_cum : matplotlib Figure/Axis for cumulative plot
    fig_pca, axes_pca : Figure/Axis array for PC1–PC2 scatters
    """
    rng = np.random.default_rng(seed)
    # -- Real PCA --
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack')
    orig_vr  = adata.uns['pca']['variance_ratio']
    orig_cum = np.cumsum(orig_vr)
    orig_pca = adata.obsm['X_pca'][:, :2]  # first two PCs

    # -- Permutations --
    perm_cums = []
    example_pcas = []  # to store first n_examples embeddings
    for i in range(n_perms):
        ad = adata.copy()
        X = ad.X.toarray() if sp.issparse(ad.X) else ad.X.copy()

        # choose & shuffle only n_perm_genes
        cols = rng.choice(X.shape[1], size=n_perm_genes, replace=False)
        for gi in cols:
            rng.shuffle(X[:, gi])  # each X[:, gi] is shuffled within (gi is one column of choice) 
        ad.X = X

        # PCA
        sc.tl.pca(ad, n_comps=n_pcs, svd_solver='arpack')
        perm_vr = ad.uns['pca']['variance_ratio']
        perm_cums.append(np.cumsum(perm_vr))

        # keep the first n_examples PCAs
        if i < n_examples:
            example_pcas.append(ad.obsm['X_pca'][:, :2])

    perm_arr   = np.vstack(perm_cums)
    mean_perm  = perm_arr.mean(axis=0)
    std_perm   = perm_arr.std(axis=0)
    pcs        = np.arange(1, n_pcs+1)

    # ----- 1) cumulative variance plot -----
    fig_cum, ax_cum = plt.subplots(figsize=(6,4))
    ax_cum.fill_between(pcs,
                        mean_perm - std_perm,
                        mean_perm + std_perm,
                        alpha=0.3,
                        label=f'Permuted ±1 std\n({n_perm_genes} genes)',
                        zorder=1)
    ax_cum.plot(pcs, mean_perm,    marker='o', linestyle='--',
                label='Permuted mean', zorder=3)
    ax_cum.plot(pcs, orig_cum,     marker='o',
                label='Original',       zorder=4)
    
    ax_cum.axhline(orig_cum[10], linestyle=':', color='gray', label=f'{np.round(orig_cum[10]*100)}% thresh', zorder=2)
    ax_cum.set(
        xlabel='Number of PCs',
        ylabel='Cumulative explained variance',
        title=f'Permute {n_perm_genes} genes — {n_perms} runs'
    )
    ax_cum.legend(frameon=False)
    fig_cum.tight_layout()

    # ----- 2) PC1–PC2 scatter examples -----
    n_cols = 1 + len(example_pcas)
    fig_pca, axes = plt.subplots(
    1,
    n_cols,
    figsize=(4 * n_cols, 4),
    squeeze=False)

    axes_pca = axes[0]
    axes_pca[0].scatter(orig_pca[:,0], orig_pca[:,1],
                        s=5, alpha=0.7, cmap=cmap_original,
                        c='gray', label='Original')
    axes_pca[0].set_title('Original PCA')
    axes_pca[0].set_xlabel('PC1')
    axes_pca[0].set_ylabel('PC2')
    axes_pca[0].set_aspect('equal')

    # permuted examples
    for j, perm_pca in enumerate(example_pcas, start=1):
        axes_pca[j].scatter(perm_pca[:,0], perm_pca[:,1],
                            s=5, alpha=0.7, cmap=cmap_perm,
                            c='gray', label=f'Perm #{j}')
        axes_pca[j].set_title(f'Permuted PCA #{j}')
        axes_pca[j].set_xlabel('PC1')
        axes_pca[j].set_ylabel('PC2')
        axes_pca[j].set_aspect('equal')

    fig_pca.tight_layout()
    return (fig_cum, ax_cum), (fig_pca, axes_pca)





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
    plt.show()
