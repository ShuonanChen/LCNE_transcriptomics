# config.py
import os, sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============= PATH CONFIGURATION =============
# Determine paths based on Code Ocean environment
# In reproducible runs: /code is working directory, /data and /results are mounted
_CODE_OCEAN = os.path.exists('/code') and os.path.exists('/data') and os.path.exists('/results')

if _CODE_OCEAN:
    # Code Ocean environment
    PROJECT_ROOT = '/code/'
    DATA_DIR = '/data/'
    # OUTPUT_DIR = '/results/'
    OUTPUT_DIR = '/results/'
else:
    # Local/IDE environment
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Add the code directory to Python path so modules can be imported
CODE_DIR = PROJECT_ROOT if _CODE_OCEAN else os.path.join(PROJECT_ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# Define standard paths
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")


# Sub-directories for different data types
SNRNA_DATA_DIR = os.path.join(DATA_DIR, "LCNE-transcriptomics-preprocessing_2026-07-08_16-23-00/snRNAseq/")
MERFISH_DATA_DIR = os.path.join(DATA_DIR, "LCNE-transcriptomics-preprocessing_2026-07-08_16-23-00/merfish/")
RETROSEQ_DATA_DIR = os.path.join(DATA_DIR, "LCNE-transcriptomics-preprocessing_2026-07-08_16-23-00/retroseq/")
MESH_DIR = os.path.join(DATA_DIR, "LC_percentile_meshes/")
# MESH_DIR_sym = os.path.join(DATA_DIR, "mesh/")
TMP_OUT_DIR = OUTPUT_DIR

# Data files provided as data assets (see code/make_gencode_metadata.py and
# code/make_mmidas_metadata.py). Code Ocean mounts each asset under a directory named
# after the *asset* (not the file), so the mount folder name is not predictable.
# _resolve_data_file finds the file by name whether it is loose in /data or nested one
# level inside any asset-mount directory.
import glob as _glob


def _resolve_data_file(filename):
    """Locate a data file by name: loose in /data, or nested one level inside a
    data-asset mount directory (whatever the mount is named). Falls back to the
    direct /data path if not found."""
    direct = os.path.join(DATA_DIR, filename)
    if os.path.exists(direct):
        return direct
    hits = sorted(_glob.glob(os.path.join(DATA_DIR, "*", filename)))
    return hits[0] if hits else direct


# GENCODE vM38 mouse gene annotation (retroseq gene-length -> TPM); no runtime download.
GENCODE_GTF = _resolve_data_file("gencode.vM38.annotation.gtf.gz")
# MMIDAS clustering-outcome pickle (used by snRNA/continuum_analayis.ipynb).
MMIDAS_PKL = _resolve_data_file("all_mmidas_outcome_w_seed_w_pca.pkl")

# Output directories for figures
SNRNA_FIGURE_DIR = os.path.join(FIGURE_DIR, "snRNAseq")
MERFISH_FIGURE_DIR = os.path.join(FIGURE_DIR, "merfish")
RETROSEQ_FIGURE_DIR = os.path.join(FIGURE_DIR, "retroseq")

# lets use actual CPM 
# CPM_SCL = 1e4 # CPM default set to be 1e4 for now. (so count per 10k really)
CPM_SCL = 1e6
CPM_SCL_MERFISH = 1 # the load data is already normalized to have 1000 counts per cell!!!

CMAP_NAME = 'PiYG'  # used for the pseudocluster only for now!


# Create necessary directories
for directory in [DATA_DIR, OUTPUT_DIR, FIGURE_DIR, 
                 SNRNA_DATA_DIR, MERFISH_DATA_DIR, RETROSEQ_DATA_DIR, MESH_DIR,
                 SNRNA_FIGURE_DIR, MERFISH_FIGURE_DIR, RETROSEQ_FIGURE_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============= PLOT CONFIGURATION =============
# Default font path
FONT_PATH = os.path.join(CODE_DIR, 'utils', 'Helvetica.ttc')


def configure_matplotlib():
    """Configure matplotlib for publication-quality figures"""
    plt.rcParams.update({
        "svg.fonttype": 'none',
        "pdf.fonttype": 42,
        'ps.fonttype': 42,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8
    })
    
    # Set font if available
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Helvetica']
    else:
        print("Warning: Helvetica font not found. Using default font.")

# ============= HELPER FUNCTIONS =============
def save_figure(filename, dir_path=FIGURE_DIR, formats=["svg", "png"], dpi=500,
                source_data=None, verify=True, strict=False):
    """
    Save figure in multiple formats, optionally writing (and verifying) source data.
    -----------
    formats : list, default=["svg", "png"]
        List of formats to save figure in
    dpi : int, default=300
        Resolution for raster formats
    source_data : pd.DataFrame        -> writes <filename>_source_data.csv
                : dict{panel: pd.DataFrame} -> writes ONE tidy <filename>_source_data.csv
                                               with a 'panel' column
                : None                -> no source data written
    verify : bool, default=True
        After writing, re-load the CSV and check its numbers against the numbers
        actually plotted in the current figure (plt.gcf()). Reports PASS/FAIL/WARN.
    strict : bool, default=False
        If True, raise AssertionError on any verification FAIL. Default only reports.
    """
    os.makedirs(dir_path, exist_ok=True)

    # Convert single format to list
    if isinstance(formats, str):
        formats = [formats]

    # Save in each format
    for format in formats:
        full_path = f"{dir_path}/{filename}.{format}"
        print(f"Saving figure to: {full_path}")
        plt.savefig(full_path, format=format, dpi=dpi)

    # Write publication source data alongside the figure
    if source_data is not None:
        _save_source_data(filename, dir_path, source_data)
        if verify:
            try:
                verify_source_data(plt.gcf(), source_data, filename=filename,
                                   dir_path=dir_path, from_file=True, strict=strict)
            except AssertionError:
                raise
            except Exception as e:  # never let verification break a figure save
                print(f"[verify] SKIPPED for {filename}: {type(e).__name__}: {e}")


def _save_source_data(filename, dir_path, source_data):
    """Write publication source data as CSV. A single DataFrame -> <filename>_source_data.csv
    (index preserved). A dict{panel: DataFrame} -> ONE tidy CSV with a leading 'panel' column
    and the per-panel index carried in a 'cell_id' column (columns are outer-joined across
    heterogeneous panels, NaN where a column does not apply)."""
    import pandas as pd

    out_path = f"{dir_path}/{filename}_source_data.csv"
    if isinstance(source_data, pd.DataFrame):
        source_data.to_csv(out_path)
    elif isinstance(source_data, dict):
        frames = {str(k): v for k, v in source_data.items()}
        tidy = pd.concat(frames, names=["panel"])
        tidy.index = tidy.index.set_names(["panel", "cell_id"])
        tidy = tidy.reset_index()
        tidy.to_csv(out_path, index=False)
    else:
        raise TypeError(
            "source_data must be a pandas DataFrame or dict of DataFrames, "
            f"got {type(source_data)}")
    print(f"Saving source data to: {out_path}")


# ---- source-data <-> plotted-artist verification -----------------------------
_COORD_NAMES = {"PCA1", "PCA2", "UMAP1", "UMAP2", "A-P", "D-V", "M-L", "X", "Y", "x", "y"}


def _load_source_data_csv(path):
    """Re-load a written source-data CSV back into {panel: DataFrame} (or {'': df})."""
    import pandas as pd
    header = pd.read_csv(path, nrows=0)
    if "panel" in header.columns:                 # tidy dict-CSV (written index=False)
        df = pd.read_csv(path)
        panels = {}
        for name, g in df.groupby("panel", sort=False):
            g = g.drop(columns=["panel"])
            if "cell_id" in g.columns:
                g = g.set_index("cell_id")
            panels[str(name)] = g.dropna(axis=1, how="all")
        return panels
    # single-DataFrame CSV was written with its index as column 0 -> restore it
    return {"": pd.read_csv(path, index_col=0)}


def _cost(a, b):
    """max|Δ| between two 1-D arrays compared as sorted multisets (NaN-aware).
    Returns inf if lengths (or finite-value counts) differ."""
    import numpy as np
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        return np.inf
    sa = np.sort(a); sb = np.sort(b)          # NaN sorts to the end
    fa = sa[~np.isnan(sa)]; fb = sb[~np.isnan(sb)]
    if fa.size != fb.size:                    # differing NaN counts -> not comparable
        return np.inf
    if fa.size == 0:
        return 0.0
    return float(np.max(np.abs(fa - fb)))


def _extract_artists(fig):
    """Collect plotted numeric arrays from a figure, excluding mesh/background/colorbar
    artists. Returns (value_pool, matrix_pool, curve_pool, bg_counts).
    value_pool / matrix_pool entries are dicts {vals, n, kind, artist, used=False};
    curve_pool entries carry the full (N,2) xy array."""
    import numpy as np
    from matplotlib.collections import (PathCollection, QuadMesh, LineCollection,
                                         PolyCollection)
    try:
        from matplotlib.collections import TriMesh
    except Exception:                          # older mpl
        TriMesh = ()
    from matplotlib.patches import Rectangle

    value_pool, matrix_pool, curve_pool = [], [], []
    bg = {"mesh": 0, "gray_scatter": 0, "colorbar": 0, "constant_line": 0}

    def _is_gray_mesh(coll):
        try:
            fc = np.asarray(coll.get_facecolor())
            sizes = np.asarray(coll.get_sizes())
            al = coll.get_alpha()
            al = 1.0 if al is None else (al if np.isscalar(al) else np.max(al))
            if fc.size == 0:
                return False
            rgb = fc[0][:3]
            gray = abs(rgb[0] - rgb[1]) < 0.05 and abs(rgb[1] - rgb[2]) < 0.05
            small = (sizes.size == 0) or (np.max(sizes) <= 2)
            return gray and small and al < 0.15
        except Exception:
            return False

    for ax in fig.axes:
        for coll in ax.collections:
            if isinstance(coll, (LineCollection, PolyCollection)) or (TriMesh and isinstance(coll, TriMesh)):
                bg["mesh"] += 1
                continue
            if isinstance(coll, QuadMesh):
                arr = np.ma.asarray(coll.get_array())
                shp = arr.shape
                flat = np.asarray(arr, dtype=float).ravel()
                # colorbar ramp: one dim == 1 and monotonic
                if len(shp) == 2 and min(shp) == 1:
                    d = np.diff(flat)
                    if np.all(d >= -1e-12) or np.all(d <= 1e-12):
                        bg["colorbar"] += 1
                        continue
                matrix_pool.append(dict(vals=flat, n=flat.size, kind="matrix",
                                        artist="QuadMesh", used=False))
                continue
            if isinstance(coll, PathCollection):
                if _is_gray_mesh(coll):
                    bg["gray_scatter"] += 1
                    continue
                offs = np.asarray(coll.get_offsets(), dtype=float)
                arr = coll.get_array()
                if arr is not None:
                    a = np.asarray(arr, dtype=float).ravel()
                    value_pool.append(dict(vals=a, n=a.size, kind="color",
                                           artist="PathCollection", used=False))
                if offs.ndim == 2 and offs.shape[0] > 0:
                    for j in range(offs.shape[1]):
                        value_pool.append(dict(vals=offs[:, j], n=offs.shape[0],
                                               kind="offset", artist="PathCollection",
                                               used=False))
        # images (imshow) -> matrix
        for im in ax.images:
            a = np.asarray(im.get_array(), dtype=float).ravel()
            matrix_pool.append(dict(vals=a, n=a.size, kind="matrix",
                                    artist="AxesImage", used=False))
        # bars, per container so two series stay separate
        for cont in getattr(ax, "containers", []):
            heights = [p.get_height() for p in cont if isinstance(p, Rectangle)]
            if heights:
                h = np.asarray(heights, dtype=float)
                value_pool.append(dict(vals=h, n=h.size, kind="bar",
                                       artist="BarContainer", used=False))
        # line curves (>2 pts, non-constant); 2-pt lines are axhline/scalebars
        for ln in ax.lines:
            xy = ln.get_xydata()
            if xy is None or len(xy) < 3:
                bg["constant_line"] += 1 if (xy is not None and len(xy) == 2) else 0
                continue
            curve_pool.append(dict(xy=np.asarray(xy, dtype=float), n=len(xy),
                                   kind="curve", artist="Line2D", used=False))
    return value_pool, matrix_pool, curve_pool, bg


def _best_match(vec, pool, rtol, atol):
    """Find the unused pool entry of equal length with smallest cost. Returns
    (entry, cost, passed) or (None, None, None) if no equal-length candidate."""
    import numpy as np
    best, best_cost = None, np.inf
    had_candidate = False
    for e in pool:
        if e["used"] or e["n"] != len(vec):
            continue
        had_candidate = True
        c = _cost(vec, e["vals"])
        if c < best_cost:
            best, best_cost = e, c
    if not had_candidate:
        return None, None, None
    # pass decision uses allclose on sorted finite values
    a = np.sort(np.asarray(vec, dtype=float)); b = np.sort(np.asarray(best["vals"], dtype=float))
    fa = a[~np.isnan(a)]; fb = b[~np.isnan(b)]
    passed = (fa.size == fb.size) and np.allclose(fa, fb, rtol=rtol, atol=atol)
    return best, best_cost, passed


def verify_source_data(fig, source_data, *, filename=None, dir_path=FIGURE_DIR,
                       from_file=True, rtol=1e-5, atol=1e-8, strict=False, verbose=True):
    """Verify that a written source-data CSV contains exactly the numbers plotted in `fig`.

    Extracts numeric arrays from the live matplotlib artists and matches each source-data
    value column (order-invariant, as sorted multisets) to a plotted artist. Value / matrix /
    curve columns are HARD-checked (PASS/FAIL). Coordinate columns are soft (PASS if they
    happen to match un-jittered offsets, else WARN — jittered/mirrored spatial coords differ
    from the stored coords). Categorical columns and derived plots (violin/hist, which expose
    no per-point artist) are WARN-only. Returns a report dict; raises only if strict and a FAIL.
    """
    import numpy as np, pandas as pd, os, json

    # Normalise source into {panel: DataFrame}
    if from_file and filename is not None:
        path = f"{dir_path}/{filename}_source_data.csv"
        panels = _load_source_data_csv(path)
    elif isinstance(source_data, dict):
        panels = {str(k): v for k, v in source_data.items()}
    else:
        panels = {"": source_data}

    value_pool, matrix_pool, curve_pool, bg = _extract_artists(fig)
    rows = []

    def _num_cols(df):
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for panel, df in panels.items():
        numeric = _num_cols(df)
        value_cols = [c for c in numeric if str(c) not in _COORD_NAMES]
        coord_cols = [c for c in numeric if str(c) in _COORD_NAMES]
        cat_cols = [c for c in df.columns if c not in numeric]

        # (1) matrix panel: all-numeric grid matching a matrix artist by total size
        matched_matrix = False
        if numeric and not coord_cols and len(numeric) >= 2:
            block = df[numeric].to_numpy(dtype=float).ravel()
            e, cost, passed = _best_match(block, matrix_pool, rtol, atol)
            if e is not None:
                e["used"] = True
                matched_matrix = True
                rows.append((panel, "<matrix>", "matrix", e["artist"], len(block),
                             cost, "PASS" if passed else "FAIL"))
        if matched_matrix:
            continue

        # (2) value columns -> hard match against scatter color/offset/bar arrays
        for c in value_cols:
            vec = df[c].to_numpy(dtype=float)
            e, cost, passed = _best_match(vec, value_pool, rtol, atol)
            if e is None:
                rows.append((panel, c, "value", "-", len(vec), None,
                             "WARN(no-artist)"))
            else:
                e["used"] = True
                rows.append((panel, c, "value", e["artist"], len(vec), cost,
                             "PASS" if passed else "FAIL"))

        # (3) coordinate columns -> soft (curve if this panel is coords-only)
        if coord_cols and not value_cols:
            xy = df[coord_cols].to_numpy(dtype=float)
            matched = False
            for e in curve_pool:
                if e["used"] or e["n"] != len(xy):
                    continue
                c0 = _cost(xy[:, 0], e["xy"][:, 0]); c1 = _cost(xy[:, 1], e["xy"][:, 1])
                if max(c0, c1) < np.inf:
                    passed = c0 <= atol + rtol * np.nanmax(np.abs(xy)) and \
                             c1 <= atol + rtol * np.nanmax(np.abs(xy))
                    e["used"] = True; matched = True
                    rows.append((panel, "+".join(map(str, coord_cols)), "curve",
                                 "Line2D", len(xy), max(c0, c1),
                                 "PASS" if passed else "FAIL"))
                    break
            if not matched:
                rows.append((panel, "+".join(map(str, coord_cols)), "coord", "-",
                             len(xy), None, "WARN(coords)"))
        else:
            for c in coord_cols:
                vec = df[c].to_numpy(dtype=float)
                e, cost, passed = _best_match(vec, value_pool, rtol, atol)
                if e is not None and passed:
                    e["used"] = True
                    rows.append((panel, c, "coord", e["artist"], len(vec), cost, "PASS"))
                else:
                    rows.append((panel, c, "coord", "-", len(vec), None, "WARN(coords)"))

        # (4) categorical columns -> provenance only
        for c in cat_cols:
            rows.append((panel, c, "categorical", "-", len(df), None, "WARN(provenance)"))

    unmatched = sum(1 for e in value_pool + matrix_pool if not e["used"])
    n_fail = sum(1 for r in rows if r[6] == "FAIL")
    n_pass = sum(1 for r in rows if r[6] == "PASS")
    n_warn = sum(1 for r in rows if str(r[6]).startswith("WARN"))
    result = "FAIL" if n_fail else "PASS"

    report = dict(figure=filename, result=result,
                  n_pass=n_pass, n_fail=n_fail, n_warn=n_warn,
                  ignored_background=bg, unmatched_artists=unmatched,
                  panels=[dict(panel=p, column=str(c), kind=k, artist=a,
                               n=n, max_abs_diff=(None if d is None else float(d)),
                               status=s) for (p, c, k, a, n, d, s) in rows])

    if verbose:
        print(f"[verify] {filename}: {result}  "
              f"(PASS={n_pass} FAIL={n_fail} WARN={n_warn}; "
              f"bg={bg}; unmatched_artists={unmatched})")
        for (p, c, k, a, n, d, s) in rows:
            dd = "" if d is None else f"{d:.2e}"
            print(f"    [{s:16}] {str(p)[:22]:22} {str(c)[:26]:26} {k:11} {str(a):14} n={n} {dd}")

    # append to an aggregated JSONL alongside the figures
    try:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        with open(os.path.join(FIGURE_DIR, "source_data_verification.jsonl"), "a") as fh:
            fh.write(json.dumps(report) + "\n")
    except Exception as e:
        print(f"[verify] could not write report log: {e}")

    if strict and n_fail:
        raise AssertionError(f"source-data verification FAILED for {filename}: "
                             f"{n_fail} mismatched panel(s)")
    return report

# ============= PROJECT CONSTANTS =============
# Add any project-specific constants here
RANDOM_SEED = 42


# Print debug information
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"CODE_DIR: {CODE_DIR}")
print(f"CODE_DIR exists: {os.path.exists(CODE_DIR)}")
print(f"Python path includes CODE_DIR: {CODE_DIR in sys.path}")