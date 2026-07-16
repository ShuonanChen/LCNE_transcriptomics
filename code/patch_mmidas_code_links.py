#!/usr/bin/env python
"""Patch the MMIDAS processing.json so its code link points at the actual notebooks.

`code/make_mmidas_metadata.py` records the MMIDAS code provenance as a single repo
URL. This post-processor narrows it to the two notebooks that produced the asset
(repo AllenInstitute/MMIDAS_LC-NE):
  - code.url  -> 00b_two_arms_multiseed.ipynb   (the MMIDAS run itself)
  - notes     -> 00a_prepare_data_for_mmidas.ipynb (input-data prep) is appended,
                 since aind-data-schema Code.url holds only ONE url.

Usage
    # 1) regenerate the metadata first (writes /results/<asset>/processing.json)
    python code/make_mmidas_metadata.py
    # 2) patch it
    python scratch/patch_mmidas_code_links.py [path/to/processing.json]

If no path is given, the script finds the staged MMIDAS processing.json under
/results. It edits the file in place and re-validates it. Idempotent: running it
again does not duplicate the notes line.

Note: this lives in scratch/ (Code Ocean scratch, ephemeral / not git-tracked).
"""

import glob
import json
import os
import sys

from aind_data_schema.core.processing import Processing

NB_00A = ("https://github.com/AllenInstitute/MMIDAS_LC-NE/blob/main/"
          "notebooks/00a_prepare_data_for_mmidas.ipynb")
NB_00B = ("https://github.com/AllenInstitute/MMIDAS_LC-NE/blob/main/"
          "notebooks/00b_two_arms_multiseed.ipynb")

NOTES_LINE = (
    "Notebooks (repo AllenInstitute/MMIDAS_LC-NE): input data prepared with "
    f"{NB_00A}; the MMIDAS two-arm multiseed run is code.url ({NB_00B})."
)


def _find_processing_json():
    """Path from argv[1], else the staged MMIDAS processing.json under /results."""
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.isfile(path):
            sys.exit(f"ERROR: no such file: {path}")
        return path
    hits = sorted(glob.glob("/results/LC-mmidas-results-seed-pca_*/processing.json"))
    if not hits:
        sys.exit("ERROR: no MMIDAS processing.json found under /results. "
                 "Run `python code/make_mmidas_metadata.py` first.")
    return hits[-1]


def main():
    path = _find_processing_json()
    out_dir = os.path.dirname(path)

    processing = Processing.model_validate(json.load(open(path)))
    dp = processing.data_processes[0]

    before = dp.code.url
    dp.code.url = NB_00B

    notes = dp.notes or ""
    if NB_00A not in notes:                       # idempotent
        dp.notes = (notes + ("\n\n" if notes else "") + NOTES_LINE)

    # write_standard_file writes "processing.json" into out_dir (re-validates)
    processing.write_standard_file(output_directory=out_dir)

    print(f"Patched: {path}")
    print(f"  code.url: {before}")
    print(f"        -> {dp.code.url}")
    print(f"  notes now references 00a: {NB_00A in (dp.notes or '')}")


if __name__ == "__main__":
    main()
