import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def _cramers_v(a, b):
    """Bias-corrected Cramer's V between two categorical Series (NaNs dropped pairwise)."""
    df = pd.DataFrame({"a": np.asarray(a), "b": np.asarray(b)}).dropna()
    if df["a"].nunique() < 2 or df["b"].nunique() < 2:
        return np.nan
    ct = pd.crosstab(df["a"], df["b"])
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    r, k = ct.shape
    phi2 = chi2 / n
    # Bergsma-Wicher bias correction
    phi2corr = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(rcorr - 1, kcorr - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2corr / denom))


def screen_metadata_for_batch(adata, bio_col="injection_site",
                              mouse_col="external_donor_name",
                              min_levels=2, max_levels=30):
    """Screen obs columns for batch-effect potential, three axes per column.

    A metadata column only warrants concern as a batch effect if it (1) is
    associated with / confounded by the biological factor ``bio_col``, (2) is
    NOT already absorbed by the ``mouse_col`` term (i.e. it varies within a
    mouse), and (3) actually structures expression (see
    ``variance_explained_by_covariate`` / ``evaluate_batch_effects_rf``).

    Candidate columns are auto-selected as categorical-like columns with
    ``min_levels <= nunique <= max_levels`` (excludes ``bio_col``, ``mouse_col``,
    all-NaN, and per-cell-ID columns). Returns a DataFrame indexed by column:
      n_levels            : number of observed levels
      cramers_v_site      : bias-corrected Cramer's V vs bio_col (assoc/confounding)
      pct_levels_one_site : fraction of the column's levels confined to a single bio level
      nested_in_mouse     : fraction of mice where the column is constant (1.0 => fully
                            absorbed by the mouse term)
      cramers_v_mouse     : Cramer's V vs mouse_col (context)
    sorted by cramers_v_site descending.
    """
    obs = adata.obs
    bio = obs[bio_col]
    # use only cells with a defined biological label (e.g. thalamus -> NaN dropped)
    bio_ok = bio.notna() & (bio.astype(str).str.lower() != "nan")
    n_obs = obs.shape[0]

    rows = {}
    for col in obs.columns:
        if col in (bio_col, mouse_col):
            continue
        s = obs[col]
        nun = s.nunique(dropna=True)
        if nun < min_levels or nun > max_levels or nun >= n_obs:
            continue

        sub = pd.DataFrame({"col": s.astype(str), "site": bio.astype(str)})[bio_ok.values]
        sub = sub[sub["site"].str.lower() != "nan"]
        ct = pd.crosstab(sub["col"], sub["site"])
        n_per_level = (ct > 0).sum(axis=1)
        pct_one_site = float((n_per_level == 1).mean()) if len(n_per_level) else np.nan

        const_per_mouse = obs.groupby(mouse_col, observed=True)[col].nunique(dropna=False)
        nested = float((const_per_mouse <= 1).mean())

        rows[col] = {
            "n_levels": int(nun),
            "cramers_v_site": _cramers_v(sub["col"], sub["site"]),
            "pct_levels_one_site": pct_one_site,
            "nested_in_mouse": nested,
            "cramers_v_mouse": _cramers_v(s.astype(str), obs[mouse_col].astype(str)),
        }

    out = pd.DataFrame.from_dict(rows, orient="index")
    if not out.empty:
        out = out.sort_values("cramers_v_site", ascending=False)
    return out


def variance_explained_by_covariate(adata, cols, use_rep="X_pca", n_pcs=20):
    """Fraction of (PCA) expression variance explained by each metadata column.

    For each column, computes a per-PC association with the embedding and
    aggregates over the first ``n_pcs`` PCs weighted by PC variance:
      * categorical column -> one-way ANOVA eta-squared per PC,
      * numeric/continuous column -> squared Pearson correlation per PC.
    This is a cheap expression-association magnitude that needs no group-aware
    CV, so it also works for columns nested in mouse. Returns a Series in [0, 1].
    """
    Z = np.asarray(adata.obsm[use_rep])
    k = min(n_pcs, Z.shape[1])
    Z = Z[:, :k]
    pc_var = Z.var(axis=0)
    pc_var = pc_var if pc_var.sum() > 0 else np.ones(k)
    w = pc_var / pc_var.sum()

    def eta2_one_pc(z, g):
        # categorical g: between-group SS / total SS
        df = pd.DataFrame({"z": z, "g": g}).dropna()
        if df["g"].nunique() < 2 or len(df) < 3:
            return np.nan
        grand = df["z"].mean()
        ss_tot = ((df["z"] - grand) ** 2).sum()
        if ss_tot == 0:
            return 0.0
        ss_bet = df.groupby("g")["z"].apply(lambda x: len(x) * (x.mean() - grand) ** 2).sum()
        return float(ss_bet / ss_tot)

    out = {}
    for col in cols:
        if col not in adata.obs.columns:
            continue
        s = adata.obs[col]
        is_numeric = pd.api.types.is_numeric_dtype(s) and not isinstance(
            s.dtype, pd.CategoricalDtype)
        vals = np.full(k, np.nan)
        for j in range(k):
            z = Z[:, j]
            if is_numeric:
                d = pd.DataFrame({"z": z, "x": pd.to_numeric(s, errors="coerce")}).dropna()
                if len(d) > 2 and d["x"].nunique() > 1:
                    r = np.corrcoef(d["z"], d["x"])[0, 1]
                    vals[j] = r ** 2
            else:
                vals[j] = eta2_one_pc(z, s.astype(str).values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[col] = float(np.nansum(w * np.nan_to_num(vals, nan=0.0)))
    return pd.Series(out, name="var_explained").sort_values(ascending=False)

def create_projection_subsets(adata, groupby_col='injection_site'):
    """Split dataset by projection target and preprocess each subset"""
    adata_dict = {}
    group_names = adata.obs[groupby_col].unique()    
    for group in group_names:
        adata_subset = adata[adata.obs[groupby_col] == group].copy()
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
        sc.tl.umap(adata_subset, random_state=210)
        ax = sc.pl.umap(adata_subset, color=color_vars, ncols=3, show=False)
        for a in ax:
            a.set_aspect('equal')
        plt.tight_layout()

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
                              use_pca=False, n_components=50,
                              scoring="balanced_accuracy", class_weight="balanced",
                              group_col=None, min_class_count=2, n_splits=5,
                              random_state=330):
    """Evaluate batch effects using Random Forest classification.

    Imbalance handling
    ------------------
    - ``scoring="balanced_accuracy"`` (mean per-class recall) does not reward
      always predicting the majority class; its chance level genuinely is
      ``1/n_classes`` for both a random and a majority-only classifier, so the
      dashed chance line in ``plot_rf_results`` is valid. We also report the
      majority-class prevalence as the honest no-information baseline (the
      relevant floor if you switch ``scoring`` back to ``"accuracy"``).
    - ``class_weight="balanced"`` reweights the forest so minority classes are
      not ignored. Pass ``None`` to disable.

    Cross-validation
    ----------------
    - ``group_col`` (e.g. ``"external_donor_name"``): when set, classes for a
      *different* target are evaluated with ``GroupKFold`` so that whole groups
      (e.g. mice) are held out. This is the honest test for a nested design:
      it asks whether a target (e.g. ``injection_site``) is separable in a way
      that *generalizes across mice*, rather than letting the model recognize
      individual cells from a mouse seen in training. Grouping is skipped (with
      a warning) when ``group_col == var``, since a class cannot be held out and
      predicted at the same time.
    - ``min_class_count``: classes (and, when grouping, groups) with fewer than
      this many cells are dropped before CV, avoiding degenerate folds (e.g. a
      mouse with a single cell).
    """
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    results = {}

    def make_classifier():
        rf = RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=-1,
                                    random_state=0, class_weight=class_weight)
        if use_pca:
            return Pipeline([("pca", PCA(n_components=n_components, random_state=0)),
                             ("rf", rf)])
        return rf

    for var in target_vars: # predict those columns
        if var not in adata.obs.columns:
            continue

        y_raw = adata.obs[var].astype(str).values
        groups_raw = (adata.obs[group_col].astype(str).values
                      if (group_col is not None and group_col in adata.obs.columns)
                      else None)

        # drop cells whose target class is too small to cross-validate reliably
        vc = pd.Series(y_raw).value_counts()
        keep = np.isin(y_raw, vc[vc >= min_class_count].index.values)
        dropped = (~keep).sum()
        if dropped:
            print(f"  [{var}] dropping {dropped} cells in classes with < "
                  f"{min_class_count} cells")
        Xv, yv = X[keep], y_raw[keep]
        gv = groups_raw[keep] if groups_raw is not None else None

        y = LabelEncoder().fit_transform(yv)
        n_classes = len(np.unique(y))
        if n_classes < 2:
            print(f"  [{var}] only {n_classes} class after filtering -- skipped")
            continue

        # class-frequency baselines (computed on the filtered set actually used)
        freqs = np.bincount(y) / len(y)
        chance_level = 1.0 / n_classes          # valid floor for balanced_accuracy
        majority_baseline = float(freqs.max())  # valid floor for plain accuracy

        # choose CV scheme
        use_group = gv is not None and group_col != var
        if gv is not None and group_col == var:
            warnings.warn(f"group_col == '{var}'; cannot hold out the group being "
                          f"predicted -- falling back to StratifiedKFold for this target.")
        if use_group:
            n_groups = len(np.unique(gv))
            eff_splits = min(n_splits, n_groups)
            if eff_splits < 2:
                print(f"  [{var}] only {n_groups} group(s) -- cannot GroupKFold, skipped")
                continue
            cv = GroupKFold(n_splits=eff_splits)
            cv_iter = cv.split(Xv, y, groups=gv)
            cv_desc = f"GroupKFold(by {group_col}, {eff_splits} folds)"
        else:
            min_class = int(np.bincount(y).min())
            eff_splits = min(n_splits, min_class)
            if eff_splits < 2:
                print(f"  [{var}] smallest class has < 2 cells -- skipped")
                continue
            cv = StratifiedKFold(n_splits=eff_splits, shuffle=True, random_state=random_state)
            cv_iter = cv
            cv_desc = f"StratifiedKFold({eff_splits} folds, shuffle)"

        scores = cross_val_score(make_classifier(), Xv, y, cv=cv_iter, scoring=scoring)

        results[var] = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'chance_level': chance_level,
            'majority_baseline': majority_baseline,
            'n_classes': n_classes,
            'scoring': scoring,
            'cv': cv_desc}
        print(f"{var} {scoring}: {scores.mean():.3f} ± {scores.std():.3f} "
              f"(chance {chance_level:.3f}, majority {majority_baseline:.3f}; {cv_desc})")

    return results

def plot_rf_results(results, title_suffix=""):
    """Plot Random Forest batch effect results.

    Draws two baselines per bar: the dashed red line is the chance level
    (1/n_classes -- the valid floor for balanced_accuracy), and the dotted
    orange line is the majority-class prevalence (the valid floor for plain
    accuracy). A bar only signals a batch effect if it clears the baseline
    appropriate to its scoring metric.
    """
    labels = list(results.keys())
    means = [results[var]['mean'] for var in labels]
    errors = [results[var]['std'] for var in labels]
    chances = [results[var]['chance_level'] for var in labels]
    majorities = [results[var].get('majority_baseline', np.nan) for var in labels]
    metric = results[labels[0]].get('scoring', 'accuracy') if labels else 'accuracy'

    x = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=errors, capsize=5, alpha=0.7)
    for xi, chance, maj in zip(x, chances, majorities):
        plt.hlines(chance, xi - 0.4, xi + 0.4, color='red', linestyles="dashed", alpha=0.8,
                   label='_nolegend_')
        if np.isfinite(maj):
            plt.hlines(maj, xi - 0.4, xi + 0.4, color='darkorange', linestyles="dotted",
                       alpha=0.9, label='_nolegend_')

    for xi, m, e in zip(x, means, errors):
        plt.text(xi, m + e + 0.02, f"{m:.2f}", ha="center", va="bottom")
    # legend proxies
    plt.plot([], [], color='red', linestyle='dashed', label='chance (1/n_classes)')
    plt.plot([], [], color='purple', linestyle='dotted', label='majority-class baseline', lw=2)
    plt.xticks(x, [label.replace('_', ' ').title() for label in labels])
    plt.ylabel(metric.replace('_', ' ').title())
    plt.ylim(0, 1.0)
    plt.legend(fontsize=8, loc='upper right')
    plt.title(f"Batch Effect Detection with Random Forest{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
