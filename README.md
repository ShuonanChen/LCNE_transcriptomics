# LC-NE Transcriptomics

Transcriptomic analysis of the mouse locus coeruleus noradrenergic (LC-NE) system,
integrating single-nucleus RNA sequencing (snRNAseq), spatial transcriptomics (MERFISH),
and retrograde-labeled sequencing (retro-seq). This capsule reproduces the computational
figures for the accompanying manuscript.

This is a [Code Ocean](https://codeocean.com) capsule: the full analysis runs end to end
with a single command, and all figures are written to `/results/figures/`.

Source code is also available on GitHub:
[AllenNeuralDynamics/LCNE_transcriptomics](https://github.com/AllenNeuralDynamics/LCNE_transcriptomics).

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

If you use this pipeline or the derived datasets, please cite the associated preprint
and the Allen Institute for Neural Dynamics.

> Su, Z., Kosillo, P., Jung, K., Chen, S., Summers, M. T., Piet, A., ... Chandrashekar,
> J. V., & Cohen, J. Y. (2026). *Topographic structure and function of locus coeruleus
> norepinephrine neurons.* bioRxiv. https://doi.org/10.64898/2026.04.10.717727

```bibtex
@article{su2026topographic,
  title     = {Topographic structure and function of locus coeruleus norepinephrine neurons},
  author    = {Su, Zhixiao and Kosillo, Polina and Jung, Kanghoon and Chen, Shuonan and Summers, Mathew T. and Piet, Alex and Hou, Han and Hagihara, Kenta M. and Friedmann, Drew and Ho-Shing, Olivia and Becker, Matthew I. and Chartrand, Thomas and Grotz, Peter and Hilton-VanOsdall, Ella and Lee, Margaret and Javeri, Rajvi and Tuggle, Samantha L. and Ouellette, Naveen and Myers, Holly and Laiton, Camilo and Wulf, Kaelin and Rohde, John and Buccino, Alessio P. and Arshadi, Cameron and Wang, Di and Seshamani, Sharmishtaa and Vasquez, Sonya and Eng, Carolyn M. and Ollerenshaw, Douglas R. and Dee, Nick and Casper, Tamara and Ho, Windy and Jungert, Matthew and Jordan, Atlas and Phillips, Elliot and Chakka, Anish Bhaswanth and Nasirova, Kamiliam and Blake, Krista and McCutcheon, Audrey and Koch, Megan and Vergara, Maria Camila and Smith, Kimberly A. and Jarsky, Tim and Lusk, Nicholas and Rue, Mara C. P. and Chen, Xiaoyin and Siegle, Joshua H. and Glaser, Adam K. and Lee, Brian R. and Svoboda, Karel and Isogai, Yoh and Chandrashekar, Jayaram V. and Cohen, Jeremiah Y.},
  journal   = {bioRxiv},
  year      = {2026},
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.64898/2026.04.10.717727},
  url       = {https://www.biorxiv.org/content/10.64898/2026.04.10.717727v1}
}
```


## License

Released under the MIT License. See [LICENSE](LICENSE.txt) for details.
Copyright (c) Allen Institute for Neural Dynamics.
