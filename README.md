# LC-NE Transcriptomics

Transcriptomic analysis of the mouse locus coeruleus noradrenergic (LC-NE) system,
integrating single-nucleus RNA sequencing (snRNAseq), spatial transcriptomics (MERFISH),
and retrograde-labeled sequencing (retro-seq). This capsule reproduces the computational
figures for the accompanying manuscript.

This is a [Code Ocean](https://codeocean.com) capsule: the full analysis runs end to end
with a single command, and all figures are written to `/results/figures/`.

## Overview

The analysis spans three complementary data modalities:

- **snRNAseq** — transcriptomic profiling of LC-NE nuclei, defining the molecular
  cell-type structure and the pseudocluster continuum.
- **MERFISH** — spatial transcriptomics mapping molecular identities back onto LC
  anatomy, including cross-modality integration (CCA) and gene imputation.
- **retro-seq** — retrograde-labeled sequencing linking transcriptomic identity to
  projection targets, including projection prediction.

Figures are grouped by modality under `/results/figures/{snRNAseq,merfish,retroseq}/`.

## Reproducing the analysis

```bash
cd code
./run
```

`run` executes the analysis notebooks in order and writes every figure to
`/results/figures/`. Ordering matters: the snRNA notebooks run first because they
generate gene lists and cluster assignments that the MERFISH and retro-seq notebooks
consume.

All input and output paths are resolved through `code/notebooks/config.py`, which detects
the Code Ocean environment (`/data`, `/results`) and falls back to a local layout
automatically, so the same notebooks run unmodified in either setting.

## Data

Preprocessed inputs are provided as Code Ocean data assets that mount under `/data/`:

| Modality / resource | Contents |
|---------------------|----------|
| snRNAseq (batch-corrected) | LC-NE single-nucleus expression matrix |
| MERFISH | LC-NE spatial transcriptomics expression matrix |
| LC percentile meshes | LC anatomical reference meshes and coordinates |

The raw-to-preprocessed pipeline that produces these inputs lives in the companion
repository `LCNE_transcriptomics_preprocessing`.

## Repository layout

```
code/
  run                      # entry point: runs the analysis notebooks in order
  notebooks/
    config.py              # data/output path configuration (Code Ocean + local)
    snRNA/                 # single-nucleus RNA-seq analysis
    merfish/               # MERFISH spatial analysis
    retroseq/              # retro-seq / projection analysis
  *.py                     # analysis modules (processing, plotting, imputation, ...)
data/                      # data assets mount point (Code Ocean)
output/ -> /results        # figures written here
```

The `plot_all_*_figs.ipynb` notebooks produce the main manuscript figures for each
modality; the remaining notebooks generate supplemental figures and supporting analyses.




## Citation

If you use this pipeline or the derived datasets, please cite the associated publication
and the Allen Institute for Neural Dynamics.
```
@article{xxxx,
  title   = {Topographic structure and function of locus coeruleus
norepinephrine neurons},
  author  = {Zhixiao Su},
  journal = {xxx},
  volume  = {xx},
  number  = {xx},
  pages   = {xxx},
  year    = {xxx},
  publisher = {xxx}
}
```


## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
Copyright (c) Allen Institute for Neural Dynamics.
