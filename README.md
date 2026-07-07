# LCNE_transcriptomics

Transcriptomics analysis for the LC-NE manuscript (snRNAseq, MERFISH, and retro-seq
downstream analysis: pseudoclusters, CCA, imputation).

This is a [Code Ocean](https://codeocean.com) capsule. The full analysis is run by
`code/run`, which executes a fixed, ordered list of Jupyter notebooks. All figures are
written to `/results/figures/`.

The preprocessed input data is produced by the companion repo
`LCNE_transcriptomics_preprocessing`. After running that repo you must attach its output
as a **Code Ocean data asset** to this capsule so it mounts under `/data/`.

---

## How to reproduce

```bash
cd code
./run
```

`run` executes the notebooks below **in order** (the order matters — the snRNA notebooks
generate intermediate gene lists / cluster assignments that the MERFISH and retro-seq
notebooks consume):

1. `notebooks/snRNA/plot_all_snRNA_figs.ipynb`   *(main snRNA figures)*
2. `notebooks/snRNA/continuum_analayis.ipynb`
3. `notebooks/snRNA/structure_analysis.ipynb`
4. `notebooks/snRNA/composite_figure.ipynb`
5. `notebooks/merfish/plot_all_merfish_figs.ipynb`   *(main MERFISH figures)*
6. `notebooks/merfish/supplemental_figure_S10_v2.ipynb`
7. `notebooks/retroseq/plot_all_retroseq_figs.ipynb`   *(main retro-seq figures)*
8. `notebooks/retroseq/projection_predict.ipynb`

All paths are resolved through `code/notebooks/config.py`, which switches between the
Code Ocean layout (`/data`, `/results`) and a local layout automatically.

---

## Data assets

### Currently attached (`.codeocean/datasets.json`)

| Mount | Contents | Used as |
|-------|----------|---------|
| `LC_NE_preprocessed` → `/data/LC_NE_preprocessed/` | `snRNAseq/snRNAseq_LCNE_BN_d4_1-5k.h5ad`, `merfish/adata_mer_subset_2_2k.h5ad`, `retroseq/retroseq_updated_filtered.h5ad` | snRNA / MERFISH / retro-seq inputs |
| `LC_percentile_meshes` → `/data/LC_percentile_meshes/` | `new_core_mesh.obj`, `percentile_10..90.obj`, `LC_points.csv` | LC anatomical meshes (`MESH_DIR`) |

These cover the **main-figure** notebooks (#1, #4 partially, #7, #8) for their primary
inputs.

---

## ⚠️ Inputs that are NOT shareable yet (action required)

The following files are read by notebooks in `run` but are **not part of any attached
data asset**. On a clean Code Ocean reproduction they will be missing and the run will
fail. They currently only exist as loose files under the local `data/` directory (the
author's working copy), not in a mounted, versioned data asset.

| # | File / path referenced | `config.py` var | Resolves to | Used by | Status |
|---|------------------------|-----------------|-------------|---------|--------|
| 1 | `snRNAseq_LCNE_BN_d4_merbar_1-5k.h5ad` | `SNRNA_DATA_DIR` | `/data/LC_NE_preprocessed/snRNAseq/` | `structure_analysis`, `plot_all_merfish_figs`, `supplemental_figure_S10_v2` | **Missing from the attached `LC_NE_preprocessed` asset** — the asset only contains the non-`merbar` `snRNAseq_LCNE_BN_d4_1-5k.h5ad`. |
| 2 | `all_mmidas_outcome_new_11seed.pkl` and `..._v2.pkl` | `DATA_DIR` | `/data/` (top level) | `continuum_analayis` | Not in any asset (~206 MB each). |
| 3 | `merfish_scvi_results/{obs_names.csv, X_scVI.npy, X_umap.npy}` | — (**hard-coded** `/root/capsule/data/merfish_scvi_results/`) | local only | `plot_all_merfish_figs` | Not in any asset **and** path is hard-coded — should go through `config.py`. |
| 4 | `250513_LC_core_67_mesh_shrunk.obj`, `LC_ccf_v1_*.obj`, `subCD_*.obj`, `subCV_*.obj` | `MESH_DIR_sym` | `/data/mesh/` | `plot_all_merfish_figs`, `supplemental_figure_S10_v2` (`get_hemi`, `make_bilateral_mesh_from_ref`) | Not in any asset. Distinct from the attached `LC_percentile_meshes`. |
| 5 | `gencode.vM38.annotation.gtf.gz` | `OTHERS_DIR` | `/data/others/` | `plot_all_retroseq_figs` (gene-length normalization) | Not in any asset. Public GENCODE annotation — can be re-downloaded, but should be pinned as an asset for reproducibility. |

> Resolved: `adata_mer_subset_2_2k_foo.h5ad` was a stray working-copy file used only by a now-removed line in `plot_all_merfish_figs`; that notebook loads the attached `MERFISH_DATA_DIR/adata_mer_subset_2_2k.h5ad`, and the loose file has been deleted.

**Recommended fix:** add files 1–4 to the `LC_NE_preprocessed` asset (or a new asset)
at the paths `config.py` expects, re-attach in `.codeocean/datasets.json`, and for file
3 replace the hard-coded `/root/capsule/...` path with `MERFISH_DATA_DIR`/`DATA_DIR`.
For file 5, attach the pinned GENCODE vM38 GTF.

### ⚠️ Notebooks that write into the (read-only) data mount

In a Code Ocean reproducible run `/data` is **read-only**; these writes will fail and
should be redirected to `/results` or `/scratch`:

| Notebook | Writes to | Note |
|----------|-----------|------|
| `plot_all_retroseq_figs` | `RETROSEQ_DATA_DIR/retro_BN_d4_1500genes.h5ad` | intermediate; redirect to `/results` or `/scratch` |
| `projection_predict` | `RETROSEQ_DATA_DIR/retroseq_updated_filtered.h5ad` | **overwrites its own input** — must not write back to the asset |
| `projection_predict` | `RETROSEQ_DATA_DIR/retroseq_prediction_results.pkl` | intermediate; redirect to `/results` or `/scratch` |

---

## Per-notebook inputs and outputs

Paths use the `config.py` variables. ✅ = covered by an attached asset, ❌ = not
shareable yet (see table above), ↪ = intermediate file produced by an earlier notebook
in the run (written to `/results`, i.e. `TMP_OUT_DIR`).

### snRNA

**1. `snRNA/plot_all_snRNA_figs.ipynb`** — main snRNA figures
- Inputs: `SNRNA_DATA_DIR/snRNAseq_LCNE_BN_d4_1-5k.h5ad` ✅
- Outputs: figures → `SNRNA_FIGURE_DIR`; `cellID_pc_0722.csv` ↪; `allyourgenes.pkl` ↪

**2. `snRNA/continuum_analayis.ipynb`**
- Inputs: `SNRNA_DATA_DIR/snRNAseq_LCNE_BN_d4_1-5k.h5ad` ✅; `DATA_DIR/all_mmidas_outcome_new_11seed.pkl` ❌; `..._v2.pkl` ❌; `cellID_pc_0722.csv` ↪
- Outputs: figures (e.g. `mmidas_cluster_v2`) → `SNRNA_FIGURE_DIR`

**3. `snRNA/structure_analysis.ipynb`**
- Inputs: `SNRNA_DATA_DIR/snRNAseq_LCNE_BN_d4_1-5k.h5ad` ✅; `SNRNA_DATA_DIR/snRNAseq_LCNE_BN_d4_merbar_1-5k.h5ad` ❌; `MERFISH_DATA_DIR/adata_mer_subset_2_2k.h5ad` ✅; `cellID_pc_0722.csv` ↪
- Outputs: PCA / structure figures → `SNRNA_FIGURE_DIR`

**4. `snRNA/composite_figure.ipynb`**
- Inputs: `SNRNA_DATA_DIR/snRNAseq_LCNE_BN_d4_1-5k.h5ad` ✅
- Outputs: `composite_all_cats.{pdf,png}` and 300/600 dpi variants → output dir

### MERFISH

**5. `merfish/plot_all_merfish_figs.ipynb`** — main MERFISH figures
- Inputs: `MESH_DIR` (`new_core_mesh.obj`) ✅; `MERFISH_DATA_DIR/adata_mer_subset_2_2k.h5ad` ✅; `merfish_scvi_results/{obs_names.csv,X_scVI.npy,X_umap.npy}` ❌ (hard-coded path); `MESH_DIR_sym` meshes ❌; `SNRNA_DATA_DIR/snRNAseq_LCNE_BN_d4_merbar_1-5k.h5ad` ❌; `allyourgenes.pkl` ↪
- Outputs: CCA / imputation / heatmap figures → `MERFISH_FIGURE_DIR`

**6. `merfish/supplemental_figure_S10_v2.ipynb`**
- Inputs: `MESH_DIR` ✅; `MESH_DIR_sym` meshes ❌; `MERFISH_DATA_DIR/adata_mer_subset_2_2k.h5ad` ✅; `SNRNA_DATA_DIR/snRNAseq_LCNE_BN_d4_merbar_1-5k.h5ad` ❌
- Outputs: supplemental figure S10 (`.pdf`/`.png`, 600 dpi)

### retro-seq

**7. `retroseq/plot_all_retroseq_figs.ipynb`** — main retro-seq figures
- Inputs: `RETROSEQ_DATA_DIR/retroseq_updated_filtered.h5ad` ✅; `SNRNA_DATA_DIR/snRNAseq_LCNE_BN_d4_1-5k.h5ad` ✅; `OTHERS_DIR/gencode.vM38.annotation.gtf.gz` ❌; `cellID_pc_0722.csv` ↪
- Outputs: figures → `RETROSEQ_FIGURE_DIR`; ⚠️ writes `retro_BN_d4_1500genes.h5ad` into the read-only data mount (see warning above)

**8. `retroseq/projection_predict.ipynb`**
- Inputs: `RETROSEQ_DATA_DIR/retroseq_updated_filtered.h5ad` ✅
- Outputs: `confusions` figure → `RETROSEQ_FIGURE_DIR`; ⚠️ writes `retroseq_updated_filtered.h5ad` (overwrites input) and `retroseq_prediction_results.pkl` into the read-only data mount (see warning above)

---

## Repository layout

```
code/
  run                      # entry point: runs the 8 notebooks in order
  notebooks/
    config.py              # all data/output path configuration
    snRNA/                 # snRNA notebooks (4)
    merfish/               # MERFISH notebooks (2)
    retroseq/              # retro-seq notebooks (2)
  *.py                     # analysis modules (processing, plotting, imputations, ...)
data/                      # data assets mount point (Code Ocean)
output/ -> /results        # figures written here
```

`plot_all_*_figs.ipynb` are the main notebooks that create the manuscript's main
figures; the remaining notebooks generate supplemental figures and intermediate
artifacts.
