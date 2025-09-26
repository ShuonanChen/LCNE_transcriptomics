"""
Module for Canonical Correlation Analysis on spatial transcriptomics data.
This module provides functions to perform CCA between gene expression and spatial coordinates,
visualize results, and identify genes correlated with spatial directions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def perform_cca(X, S, n_components=2, scl_multiplier=25):
    """
    Perform Canonical Correlation Analysis between gene expression and spatial data
    
    Parameters:
    -----------
    X : numpy.ndarray
        Gene expression matrix
    S : numpy.ndarray
        Spatial coordinates
    n_components : int, default=2
        Number of components for CCA
    scl_multiplier : float, default=25
        Scaling factor for spatial coordinates
    
    Returns:
    --------
    cca : CCA object
        Fitted CCA model
    X_c : numpy.ndarray
        Transformed gene expression data
    S_c : numpy.ndarray
        Transformed spatial data
    canonical_correlations : list
        List of canonical correlations
    """
    # Scale gene expression data
    scaler_X = StandardScaler().fit(X)
    X_scaled = scaler_X.transform(X)
    
    # Scale spatial data
    scaler_S = StandardScaler().fit(S * scl_multiplier)
    S_scaled = scaler_S.transform(S)
    
    # Perform CCA
    cca = CCA(n_components=n_components)
    cca.fit(X_scaled, S_scaled)
    X_c, S_c = cca.transform(X_scaled, S_scaled)
    
    # Calculate canonical correlations
    canonical_correlations = [np.corrcoef(X_c[:, i], S_c[:, i])[0, 1] for i in range(n_components)]
    
    return cca, X_c, S_c, canonical_correlations


def visualize_cca_component(cca, X_c, canonical_correlations, spatial_coords, mesh, component=0, scale_factor=25):
    """
    Visualize a specific CCA component
    
    Parameters:
    -----------
    cca : CCA object
        Fitted CCA model
    X_c : numpy.ndarray
        Transformed gene expression data
    canonical_correlations : list
        List of canonical correlations
    spatial_coords : numpy.ndarray
        Spatial coordinates
    mesh : trimesh object
        Mesh for visualization
    component : int, default=0
        Component to visualize
    scale_factor : int, default=25
        Scale factor for visualization
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Setup the vector for visualization
    vector = cca.y_weights_[:, component]
    vector_scaled = vector * scale_factor
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(10, 5),
        gridspec_kw={'width_ratios': [1, 4]}
    )
    
    # First plot - sagittal view
    ax1.triplot(mesh.vertices.T[0], mesh.vertices.T[1], mesh.faces, color='lime', alpha=0.4)
    ax1.scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=X_c[:, component], 
                cmap='Greys', s=10, edgecolor='k', linewidth=0.1)
    ax1.set_aspect('equal')
    
    # Add arrow
    start_point_all = np.min(spatial_coords, 0)
    start_point = (start_point_all[0], start_point_all[1])
    end_point = (start_point_all[0] + vector_scaled[0], start_point_all[1] + vector_scaled[1])
    
    ax1.annotate('',
                 xy=end_point,   # arrow tip
                 xytext=start_point,  # arrow tail
                 arrowprops=dict(arrowstyle="->", color="magenta", lw=2))
    ax1.text(start_point[0], start_point[1], 
             np.around(canonical_correlations[component], 3).astype(str), 
             color='magenta')
    
    ax1.invert_yaxis()
    ax1.grid(False)
    ax1.set_ylabel("D-V axis ($\mu$m)")
    ax1.set_xlabel("A-P axis ($\mu$m)")
    
    # Format axis ticks
    xt = ax1.get_xticks(); yt = ax1.get_yticks()
    ax1.set_xticks(xt); ax1.set_yticks(yt)
    ax1.set_xticklabels([f"{int(x*25)}" for x in xt])
    ax1.set_yticklabels([f"{int(y*25)}" for y in yt])
    
    # Add scale bar
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    x0 = xlim[0] + 0.1 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax1.plot([x0, x0 + 40], [y0, y0], color='green', linewidth=3)
    ax1.text(x0 + 20, y0-2, f'{int(40*25/1000)} mm', ha='center', va='top', color='green')
    
    # Second plot - coronal view
    ax2.triplot(mesh.vertices.T[2], mesh.vertices.T[1], mesh.faces, alpha=0.4, color='lime')
    sca = ax2.scatter(
        spatial_coords[:, 2],
        spatial_coords[:, 1],
        c=X_c[:, component],
        cmap='Greys',
        s=10,
        edgecolor='k',
        linewidth=0.1)
    
    ax2.set_aspect('equal')
    
    # Add arrow to coronal view
    start_point = (start_point_all[2], start_point_all[1])
    end_point = (start_point_all[2] + vector_scaled[2], start_point_all[1] + vector_scaled[1])
    
    ax2.annotate('',
                 xy=end_point,   # arrow tip
                 xytext=start_point,  # arrow tail
                 arrowprops=dict(arrowstyle="->", color="magenta", lw=2))
    ax2.text(start_point[0], start_point[1], 
             np.around(canonical_correlations[component], 3).astype(str), 
             color='magenta')
    
    ax2.invert_yaxis()
    ax2.grid(False)
    ax2.set_yticks([])
    ax2.set_xlabel("M-L axis ($\mu$m)")
    
    # Format axis ticks
    xt = ax2.get_xticks(); yt = ax2.get_yticks()
    ax2.set_xticks(xt); ax2.set_yticks(yt)
    ax2.set_xticklabels([f"{int(x*25)}" for x in xt])
    ax2.set_yticklabels([f"{int(y*25)}" for y in yt])
    
    # Add scale bar
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()
    x0 = xlim[0] + 0.1 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax2.plot([x0, x0 + 40], [y0, y0], color='green', linewidth=3)
    ax2.text(x0 + 20, y0-2, f'{int(40*25/1000)} mm', ha='center', va='top', color='green')
    
    fig.tight_layout()
    
    return fig


def find_genes_correlated_with_direction(adata, direction, top_n=20):
    """
    Find genes that correlate with a specific direction in space
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with gene expression and spatial data
    direction : numpy.ndarray
        Direction vector to correlate with
    top_n : int, default=20
        Number of top correlated genes to return
    
    Returns:
    --------
    df_results_sorted : pandas.DataFrame
        DataFrame with gene correlations, sorted by correlation magnitude
    """
    # Normalize direction
    direction_norm = direction / np.linalg.norm(direction)
    
    # Project spatial coordinates onto direction
    projected = adata.obsm['spatial'] @ direction_norm
    
    # Convert expression data to dense array if needed
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    
    # Calculate correlations
    gene_names = adata.var_names
    results = []
    for i, gene in enumerate(gene_names):
        expr = X[:, i]
        corr, pval = spearmanr(expr, projected)
        results.append((gene, corr, pval))
    
    # Create and sort DataFrame
    df_results = pd.DataFrame(results, columns=['gene', 'spearman_corr', 'p_value'])
    df_results_sorted = df_results.sort_values(by='spearman_corr', key=lambda s: s.abs(), ascending=False)
    
    return df_results_sorted.head(top_n)


def plot_umap_with_cca_component(adata, X_c, component=0, s=10):
    """
    Plot UMAP colored by CCA component score
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with UMAP coordinates
    X_c : numpy.ndarray
        Transformed gene expression data from CCA
    component : int, default=0
        Component to visualize
    s : int, default=10
        Point size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()

    sca = ax.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1], 
                     c=X_c[:, component], cmap='Reds', s=s, edgecolor='k', linewidth=0.1)
    ax.set_aspect('equal')    
    cbar = plt.colorbar(sca, ax=ax, label='Weighted gene score', shrink=0.3)

    # Remove spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    # Add UMAP axis labels
    y0, x0 = np.min(adata.obsm['X_umap'], 0)
    y0 *= 1.4
    x0 *= 0.8
    ax.arrow(x0, y0, 1, 0, head_width=0.25, head_length=0.2,
             length_includes_head=True, linewidth=1, color='k')
    ax.arrow(x0, y0, 0, 1, head_width=0.25, head_length=0.2,
             length_includes_head=True, linewidth=1, color='black')

    ax.text(x0+1.05, y0, 'UMAP 1', va='center', ha='left', fontsize=10)
    ax.text(x0, y0+1.6, 'UMAP 2', va='bottom', ha='center', fontsize=10)
    
    return fig


def plot_cca_loadings(cca, var_names, components=(0, 1), top_n=10):
    """
    Plot top gene loadings for CCA components
    
    Parameters:
    -----------
    cca : CCA object
        Fitted CCA model
    var_names : list or array-like
        Names of variables (genes)
    components : tuple, default=(0, 1)
        Components to plot
    top_n : int, default=10
        Number of top genes to display
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=(4, 6))
    
    for i, component in enumerate(components):
        # Create DataFrame with loadings
        loadings_df = pd.DataFrame({
            'gene': var_names,
            'loading': cca.x_weights_[:, component]
        })
        loadings_df['abs_loading'] = loadings_df['loading'].abs()
        loadings_sorted = loadings_df.sort_values('abs_loading', ascending=False)
        
        # Plot
        plt.subplot(len(components), 1, i+1)
        sns.barplot(x='gene', y='loading', data=loadings_sorted.head(top_n), 
                   color='r' if i == 0 else 'b')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("" if i < len(components)-1 else "Gene")
        plt.ylabel(f"Component {component+1} weight")
        if i == 0:
            plt.title(f"Top {top_n} genes")
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # This code will only run if the script is executed directly
    print("spatial_cca.py: Module for Canonical Correlation Analysis on spatial transcriptomics data")