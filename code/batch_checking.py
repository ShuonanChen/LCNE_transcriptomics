import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def create_projection_subsets(adata, groupby_col='injection_site'):
    """Split dataset by projection target and preprocess each subset"""
    adata_dict = {}
    group_names = adata.obs[groupby_col].unique()    
    for group in group_names:
        adata_subset = adata[adata.obs[groupby_col] == group].copy()
        
        # Standard preprocessing
        sc.pp.normalize_total(adata_subset, target_sum=1e6)
        sc.pp.log1p(adata_subset)
        sc.pp.scale(adata_subset, zero_center=True, max_value=10)
        sc.tl.pca(adata_subset, n_comps=50, svd_solver='arpack')
        sc.pp.neighbors(adata_subset, use_rep='X_pca')
        
        adata_dict[group] = adata_subset
        print(f"{group}: {adata_subset.shape}")
    
    return adata_dict

def plot_batch_effects(adata_dict, color_vars=["external_donor_name", "gender", 'total_counts']):
    """Plot UMAP colored by batch variables for each subset"""
    for group_name, adata_subset in adata_dict.items():
        print(f"\n{group_name}")
        sc.tl.umap(adata_subset, random_state=210)
        ax = sc.pl.umap(adata_subset, color=color_vars, ncols=3, show=False)
        for a in ax:
            a.set_aspect('equal')
        plt.tight_layout()
        plt.show()

def preprocess_full_dataset(adata):
    """Preprocess the full dataset for batch analysis"""
    adata_processed = adata.copy()
    sc.pp.normalize_total(adata_processed, target_sum=1e6)
    sc.pp.log1p(adata_processed)
    sc.pp.scale(adata_processed, zero_center=True, max_value=10)
    sc.tl.pca(adata_processed, n_comps=50, svd_solver='arpack')
    sc.pp.neighbors(adata_processed, use_rep='X_pca')
    sc.tl.umap(adata_processed, random_state=210)
    return adata_processed

def evaluate_batch_effects_rf(adata, target_vars=['gender', 'external_donor_name', 'injection_site'], 
                              use_pca=False, n_components=50):
    """Evaluate batch effects using Random Forest classification"""
    X = adata.X
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=330)    
    if use_pca:
        classifier = Pipeline([
            ("pca", PCA(n_components=n_components, random_state=0)),
            ("rf", RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0))
        ])
    else:
        classifier = RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=-1, random_state=0)

    for var in target_vars:
        if var in adata.obs.columns:
            le = LabelEncoder()
            y = le.fit_transform(adata.obs[var].values)
            scores = cross_val_score(classifier, X, y, cv=cv, scoring="accuracy")
            n_classes = len(np.unique(y))
            chance_level = 1.0 / n_classes
            
            results[var] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'chance_level': chance_level,
                'n_classes': n_classes}
            print(f"{var} accuracy: {scores.mean():.3f} ± {scores.std():.3f} (chance: {chance_level:.3f})")
    
    return results

def plot_rf_results(results, title_suffix=""):
    """Plot Random Forest batch effect results"""
    labels = list(results.keys())
    means = [results[var]['mean'] for var in labels]
    errors = [results[var]['std'] for var in labels]
    chances = [results[var]['chance_level'] for var in labels]
    
    x = np.arange(len(labels))
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, means, yerr=errors, capsize=5, alpha=0.7)
    for xi, chance in zip(x, chances):
        plt.hlines(chance, xi - 0.4, xi + 0.4, color='red', linestyles="dashed", alpha=0.8)
    
    for xi, m, e in zip(x, means, errors):
        plt.text(xi, m + e + 0.02, f"{m:.2f}", ha="center", va="bottom")
    plt.xticks(x, [label.replace('_', ' ').title() for label in labels])
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.title(f"Batch Effect Detection with Random Forest{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def run_comprehensive_batch_analysis(adata):
    """Run complete batch effect analysis pipeline"""
    projection_subsets = create_projection_subsets(adata, 'injection_site')
    print("\n *** batcheffect by mice ID for each projection target ***")
    plot_batch_effects(projection_subsets, ["external_donor_name", "gender", 'total_counts'])
    
    # print("\n3. Creating subsets by gender...")
    # gender_subsets = create_projection_subsets(adata, 'gender')
    # print("\n4. Plotting batch effects for each gender...")
    # plot_batch_effects(gender_subsets, ["external_donor_name", "injection_site", 'total_counts'])

    print("\n*** all the samples - check mice ID batch effect *** ")
    adata_processed = preprocess_full_dataset(adata)
    ax = sc.pl.umap(adata_processed, color=["external_donor_name", "gender", 'injection_site', 'total_counts'], 
                    ncols=2, show=False)
    for a in ax:
        a.set_aspect('equal')
    plt.tight_layout()
    plt.show()

    print("\n*** Random Forest evaluation (gender and mice ID as outcome) *** ")
    rf_results = evaluate_batch_effects_rf(adata)
    plot_rf_results(rf_results, " (Raw Data)")
    
    # print("\n7. Random Forest evaluation (PCA + RF)...")
    # rf_results_pca = evaluate_batch_effects_rf(adata, use_pca=True)
    # plot_rf_results(rf_results_pca, " (PCA + RF)")
     ### we not retunring anything for now
    # return {
    #     'projection_subsets': projection_subsets,
    #     # 'gender_subsets': gender_subsets,
    #     'processed_full': adata_processed,
    #     'rf_results': rf_results,
    #     # 'rf_results_pca': rf_results_pca
    # }