#!/usr/bin/env python
"""Generate AIND-standard metadata for the MMIDAS-outcome derived data asset.

The data asset ``all_mmidas_outcome_w_seed_w_pca.pkl`` is *scientist-derived data*
(the output of a prior analysis step, not raw experimental data): it holds the
outcome of an MMIDAS run (https://github.com/AllenInstitute/MMIDAS) applied to the
LC-NE single-nucleus RNA-seq matrix. Per AIND publication standards, derived data
that contributes to published results must carry ``data_description`` and
``processing`` metadata before transfer to the open-data bucket.

Provenance / classification
    The pickle is derived from a *previous* version of the snRNAseq preprocessing
    data (4845 cells; Code Ocean asset ``211c8ff2-3754-4f55-a758-4d07e8ab2cc7``),
    which is NOT the asset attached to this capsule and is not mounted here -- that
    is expected, since source_data records lineage rather than runtime mounts. The
    snRNAseq matrix pools cells across many donor mice, so this is an *aggregation
    across subjects*: no subject/procedures metadata is inherited, and the new
    metadata consists of ``data_description`` + ``processing`` only.

Running
    python code/make_mmidas_metadata.py
    -> writes data_description.json and processing.json to /results (on Code
       Ocean) or to <repo>/metadata (locally).

    Both files validate on write; a clean run means the metadata is well-formed.

TODO before the aind-open-data transfer / publication (values not recorded in
this capsule -- the MMIDAS run was performed outside it):
    * MMIDAS_VERSION / MMIDAS_COMMIT_HASH -- the exact code version used.
    * RUN_* datetimes                     -- when the MMIDAS run was performed.
    * FUNDING_GRANT_NUMBER                 -- grant number(s) to cite.
"""

import os
from datetime import datetime, timezone

import aind_data_schema.core.data_description as ds
import aind_data_schema.core.processing as ps
from aind_data_schema_models.modalities import Modality

# ============================ EDIT THESE ============================
# The MMIDAS run happened outside this capsule, so these are not recorded here.
# Fill them in before creating the shareable/publishable data asset.
# Analysis repo (not yet public, but will be). tz-aware datetimes required by schema.
MMIDAS_URL = "https://github.com/AllenInstitute/MMIDAS_LC-NE"
# Placeholder release version: the analysis code is not yet merged to the repo, so
# v1.0 is recorded now and can be repointed to the real GitHub/CO release once merged.
MMIDAS_VERSION = "v1.0"          # OR set the commit hash below instead
MMIDAS_COMMIT_HASH = None        # e.g. "89abcdef01234567" (7-60 hex chars); leave None if using version

RUN_START = datetime(2026, 7, 9, 10, 34, 0, tzinfo=timezone.utc)
RUN_END = datetime(2026, 7, 9, 10, 34, 0, tzinfo=timezone.utc)
CREATION_TIME = RUN_END                                 # asset creation time (-> name suffix)

# Funding is required by the schema (>=1 entry). Grant number omitted for now;
# add the grant number(s) before the aind-open-data transfer.
FUNDING = [ds.Funding(funder=ds.Organization.AIND, grant_number=None)]  # TODO: grant_number

# Stage the pickle next to the metadata so ONE self-contained asset (data +
# metadata) can be created from /results. Copies from the already-attached asset
# mounted under /data (no re-upload). Only runs on Code Ocean; skipped locally.
STAGE_PICKLE = True
PICKLE_NAME = "all_mmidas_outcome_w_seed_w_pca.pkl"
PICKLE_SOURCE = os.path.join("/data", PICKLE_NAME)  # CO mounts the asset as this dir
# ===========================================================================

# Source data provenance. MMIDAS was run on a *previous* version of the
# preprocessing data (4845 cells), which is NOT the asset attached to this capsule
# and is not mounted here -- that is fine, source_data/input_data record lineage,
# not runtime mounts. Referenced by Code Ocean asset ID; the human-readable asset
# name can be added later once known.
# TODO: confirm/replace this asset ID (and add a name) before publication.
SOURCE_ASSET_ID = "211c8ff2-3754-4f55-a758-4d07e8ab2cc7"
# Informative asset name. Independent of the pickle filename/mount (PICKLE_NAME), so
# renaming here does not affect how the notebook locates the data file.
ASSET_LABEL = "LC-mmidas-results-seed-pca"
INVESTIGATOR = "Shuonan Chen"
# project_name pattern forbids underscores, so the capsule's "LCNE_transcriptomics"
# is recorded here with a dash.
PROJECT_NAME = "LCNE-transcriptomics"


def _on_code_ocean():
    return os.path.exists("/code") and os.path.exists("/data") and os.path.exists("/results")


def _base_output_dir():
    """/results on Code Ocean, else <repo>/metadata."""
    if _on_code_ocean():
        return "/results"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "metadata")


def _resolve_pickle(path):
    """Resolve the pickle path. Code Ocean mounts a data asset as a directory named
    after the asset with the file nested inside; descend into it if so."""
    if os.path.isdir(path):
        nested = os.path.join(path, os.path.basename(path))
        if os.path.isfile(nested):
            return nested
    return path if os.path.isfile(path) else None


def stage_pickle(out_dir):
    """Copy the pickle next to the metadata so /results is one self-contained asset."""
    import shutil

    if not _on_code_ocean():
        print("Not on Code Ocean; skipping pickle staging "
              "(create the asset with the metadata files only, or bundle the pickle manually).")
        return
    src = _resolve_pickle(PICKLE_SOURCE)
    if src is None:
        print(f"WARNING: pickle not found at {PICKLE_SOURCE}; skipping staging. "
              "Attach the asset (or fix PICKLE_SOURCE) to bundle it.")
        return
    dst = os.path.join(out_dir, PICKLE_NAME)
    size_gb = os.path.getsize(src) / 1e9
    print(f"Staging pickle ({size_gb:.2f} GB) -> {dst} ...")
    shutil.copy2(src, dst)
    print("  done.")


def build_processing():
    """Processing metadata: one ANALYSIS DataProcess describing the MMIDAS run."""
    code = ps.Code(
        name="MMIDAS",
        url=MMIDAS_URL,
        version=MMIDAS_VERSION,
        commit_hash=MMIDAS_COMMIT_HASH,
        # MMIDAS architecture parameters that are inferable from the outcome pickle.
        parameters={"n_arms": 2, "latent_dim": 10},
        input_data=[ps.DataAsset(name=SOURCE_ASSET_ID)],
    )
    data_process = ps.DataProcess(
        # MMIDAS clustering is not a well-known operation -> ANALYSIS + explicit name.
        process_type=ps.ProcessName.ANALYSIS,
        name="mmidas_clustering",
        stage=ps.ProcessStage.ANALYSIS,
        experimenters=[INVESTIGATOR],
        start_date_time=RUN_START,
        end_date_time=RUN_END,
        code=code,
        notes=(
            "MMIDAS (Mixture Model Inference with Discrete-coupled AutoencoderS) "
            "consensus clustering of the LC-NE snRNAseq matrix. Outcome pickle holds "
            "75 results (a sweep over random seeds x cluster count K, num_pruned = K "
            "after pruning redundant categories) from a two-arm coupled autoencoder "
            "with a 10-D latent space, plus a shared 10-D PCA embedding (pca_10) used "
            "for silhouette-based cluster-quality evaluation. MMIDAS was run on 4845 "
            "cells from a previous version of the snRNAseq preprocessing data (see "
            "input_data), not the version attached to this capsule. The MMIDAS run was "
            "performed outside this capsule; only its output is stored in the asset."
        ),
    )
    return ps.Processing(data_processes=[data_process])


def build_data_description():
    """Data description for the derived asset (aggregation across subjects)."""
    return ds.DataDescription(
        name=ds.build_data_name(ASSET_LABEL, CREATION_TIME),
        source_data=[SOURCE_ASSET_ID],
        creation_time=CREATION_TIME,
        institution=ds.Organization.AIND,
        data_level=ds.DataLevel.DERIVED,
        investigators=[ds.Person(name=INVESTIGATOR)],
        project_name=PROJECT_NAME,
        modalities=[Modality.SCRNASEQ],
        license=ds.License.CC_BY_40,
        funding_source=FUNDING,
        data_summary=(
            "MMIDAS consensus-clustering outcomes for the LC-NE single-nucleus "
            "RNA-seq dataset: 75 runs over seeds x cluster-count K from a two-arm "
            "coupled autoencoder (10-D latent), plus a shared 10-D PCA embedding, "
            "used to assess clustering quality versus K."
        ),
    )


def main():
    processing = build_processing()
    data_description = build_data_description()
    # Isolate the asset in its own subfolder (named after the asset) so it holds ONLY
    # the pickle + metadata -- /results is shared with the figure pipeline, so we must
    # not create the asset from /results itself.
    out = os.path.join(_base_output_dir(), data_description.name)
    os.makedirs(out, exist_ok=True)
    # Write the two core files directly (an aggregation-across-subjects asset needs
    # no subject/procedures, so we skip the full Metadata wrapper).
    data_description.write_standard_file(output_directory=out)
    processing.write_standard_file(output_directory=out)
    if STAGE_PICKLE:
        stage_pickle(out)
    print(f"\nAsset staged at: {out}")
    print(f"  contents: {sorted(os.listdir(out))}")
    print("Create ONE data asset from THIS folder (data + metadata together), then "
          "file the aind-open-data transfer issue.")


if __name__ == "__main__":
    main()
