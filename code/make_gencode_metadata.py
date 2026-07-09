#!/usr/bin/env python
"""Generate AIND-standard metadata for the GENCODE vM38 GTF as a public data asset.

`gencode.vM38.annotation.gtf.gz` is *non-AIND external data*: the GENCODE mouse
release M38 comprehensive gene annotation, imported for the retro-seq analysis
(gene-length -> TPM normalization). Per the AIND guide, external data needs a
``data_description`` only (there is no ``processing`` -- we did not produce it).

The goal is to package the GTF together with this metadata into a single **public**
data asset so it can be attached to any capsule without re-downloading it from
GENCODE.

Running
    python code/make_gencode_metadata.py
    -> writes data_description.json AND copies the GTF into an isolated subfolder
       /results/<asset_name>/ (on Code Ocean), or <repo>/metadata/<asset_name>/
       locally. Create ONE data asset from that subfolder.

Source (recorded in data_summary):
    GENCODE mouse release M38 (GRCm39, Ensembl 115), released 09.2025.
    https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M38/gencode.vM38.annotation.gtf.gz
    Source page: https://www.gencodegenes.org/mouse/release_M38.html
"""

import os
from datetime import datetime, timezone

import aind_data_schema.core.data_description as ds
from aind_data_schema_models.modalities import Modality

# ============================ EDIT THESE ============================
GTF_URL = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/"
           "release_M38/gencode.vM38.annotation.gtf.gz")
SOURCE_PAGE = "https://www.gencodegenes.org/mouse/release_M38.html"
DOWNLOAD_DATE = "2026-07-07"   # date the GTF was downloaded (recorded in data_summary)

# creation_time = the date the data was posted (GENCODE M38 release: 09.2025; the
# day is not published, so the 1st is used). tz-aware datetime required by schema.
CREATION_TIME = datetime(2025, 9, 1, tzinfo=timezone.utc)

INVESTIGATOR = "Shuonan Chen"
ASSET_LABEL = "gencode-vM38-annotation"
GTF_NAME = "gencode.vM38.annotation.gtf.gz"

# Stage the GTF next to the metadata so ONE self-contained public asset can be
# created from /results. Copies from the loose file currently in /data.
STAGE_GTF = True
GTF_SOURCE = os.path.join("/data", GTF_NAME)
# ====================================================================


def _on_code_ocean():
    return os.path.exists("/code") and os.path.exists("/data") and os.path.exists("/results")


def _base_output_dir():
    """/results on Code Ocean, else <repo>/metadata."""
    if _on_code_ocean():
        return "/results"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "metadata")


def _resolve_gtf(path):
    """Resolve the GTF path. Code Ocean mounts a data asset as a directory named
    after the asset with the file nested inside; descend into it if so."""
    if os.path.isdir(path):
        nested = os.path.join(path, os.path.basename(path))
        if os.path.isfile(nested):
            return nested
    return path if os.path.isfile(path) else None


def stage_gtf(out_dir):
    """Copy the GTF next to the metadata so /results is one self-contained asset."""
    import shutil

    if not _on_code_ocean():
        print("Not on Code Ocean; skipping GTF staging "
              "(create the asset with the metadata file only, or bundle the GTF manually).")
        return
    src = _resolve_gtf(GTF_SOURCE)
    if src is None:
        print(f"WARNING: GTF not found at {GTF_SOURCE}; skipping staging. "
              "Fix GTF_SOURCE to bundle it.")
        return
    dst = os.path.join(out_dir, GTF_NAME)
    size_mb = os.path.getsize(src) / 1e6
    print(f"Staging GTF ({size_mb:.1f} MB) -> {dst} ...")
    shutil.copy2(src, dst)
    print("  done.")


def build_data_description():
    """Data description for the external GENCODE annotation (non-AIND external data)."""
    return ds.DataDescription(
        name=ds.build_data_name(ASSET_LABEL, CREATION_TIME),
        creation_time=CREATION_TIME,
        # GENCODE / EMBL-EBI are not in the Organization enum -> OTHER.
        institution=ds.Organization.OTHER,
        data_level=ds.DataLevel.DERIVED,          # published external data is DERIVED
        investigators=[ds.Person(name=INVESTIGATOR)],
        project_name="external data",             # AIND convention for imported data
        # No "reference/annotation" modality exists; SCRNASEQ is the transcriptomic
        # context this annotation is used in.
        modalities=[Modality.SCRNASEQ],
        # License is mandatory (enum is only MIT / CC-BY-4.0), so the nearest option
        # is recorded; GENCODE's true terms are stated in `restrictions`.
        license=ds.License.CC_BY_40,
        restrictions=(
            "GENCODE gene annotation is freely available for use without restriction; "
            "'CC-BY-4.0' is the closest match in the schema's license enum. "
            "See https://www.gencodegenes.org/pages/data_access.html"
        ),
        # No AIND funder applies; funding_source requires >=1 entry -> OTHER.
        funding_source=[ds.Funding(funder=ds.Organization.OTHER)],
        data_summary=(
            "GENCODE mouse release M38 (GRCm39, Ensembl 115) comprehensive gene "
            f"annotation (GTF). Downloaded from {GTF_URL} on {DOWNLOAD_DATE}. "
            f"Source page: {SOURCE_PAGE}"
        ),
    )


def main():
    data_description = build_data_description()
    # Isolate the asset in its own subfolder (named after the asset) so it holds ONLY
    # the GTF + metadata -- /results is shared with the figure pipeline.
    out = os.path.join(_base_output_dir(), data_description.name)
    os.makedirs(out, exist_ok=True)
    data_description.write_standard_file(output_directory=out)
    if STAGE_GTF:
        stage_gtf(out)
    print(f"\nAsset staged at: {out}")
    print(f"  contents: {sorted(os.listdir(out))}")
    print("Create ONE PUBLIC data asset from THIS folder (GTF + metadata together). "
          f"Name its mount '{GTF_NAME}' so config.GENCODE_GTF resolves after attaching.")


if __name__ == "__main__":
    main()
