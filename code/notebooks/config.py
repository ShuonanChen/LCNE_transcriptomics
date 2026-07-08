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
SNRNA_DATA_DIR = os.path.join(DATA_DIR, "LC_NE_preprocessed/snRNAseq/")
MERFISH_DATA_DIR = os.path.join(DATA_DIR, "LC_NE_preprocessed/merfish/")
RETROSEQ_DATA_DIR = os.path.join(DATA_DIR, "LC_NE_preprocessed/retroseq/")
MESH_DIR = os.path.join(DATA_DIR, "LC_percentile_meshes/")
# MESH_DIR_sym = os.path.join(DATA_DIR, "mesh/")
OTHERS_DIR = os.path.join(DATA_DIR, "others/")
TMP_OUT_DIR = OUTPUT_DIR

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
                source_data=None):
    """
    Save figure in multiple formats
    -----------
    formats : list, default=["svg", "png"]
        List of formats to save figure in
    dpi : int, default=300
        Resolution for raster formats
    source_data : pd.DataFrame        -> writes <filename>_source_data.csv
                : dict{panel_name: pd.DataFrame} -> writes <filename>_source_data.xlsx,
                                                    one sheet per panel
                : None                -> no source data written
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


def _sanitize_sheet_name(name, used):
    """Make an Excel-safe (<=31 chars, no []:*?/\\), unique sheet name."""
    import re
    safe = re.sub(r'[\[\]:\*\?/\\]', '_', str(name))[:31] or "sheet"
    base, k = safe, 1
    while safe in used:
        suffix = f"_{k}"
        safe = base[:31 - len(suffix)] + suffix
        k += 1
    used.add(safe)
    return safe


def _save_source_data(filename, dir_path, source_data):
    """Write publication source data: a DataFrame -> CSV, a dict of DataFrames
    -> one .xlsx with a sheet per panel (falls back to per-panel CSVs if no
    Excel engine is installed). The DataFrame index is always preserved."""
    import pandas as pd

    if isinstance(source_data, pd.DataFrame):
        out_path = f"{dir_path}/{filename}_source_data.csv"
        print(f"Saving source data to: {out_path}")
        source_data.to_csv(out_path)
    elif isinstance(source_data, dict):
        out_path = f"{dir_path}/{filename}_source_data.xlsx"
        try:
            used = set()
            with pd.ExcelWriter(out_path) as writer:
                for panel_name, df in source_data.items():
                    df.to_excel(writer, sheet_name=_sanitize_sheet_name(panel_name, used))
            print(f"Saving source data to: {out_path}")
        except (ImportError, ValueError) as e:
            # No Excel engine (openpyxl/xlsxwriter) available -> one CSV per panel
            print(f"Excel writer unavailable ({e}); writing per-panel CSVs instead")
            for panel_name, df in source_data.items():
                safe = str(panel_name).replace('/', '_').replace('\\', '_')
                csv_path = f"{dir_path}/{filename}_source_data_{safe}.csv"
                print(f"Saving source data to: {csv_path}")
                df.to_csv(csv_path)
    else:
        raise TypeError(
            "source_data must be a pandas DataFrame or dict of DataFrames, "
            f"got {type(source_data)}")

# ============= PROJECT CONSTANTS =============
# Add any project-specific constants here
RANDOM_SEED = 42


# Print debug information
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"CODE_DIR: {CODE_DIR}")
print(f"CODE_DIR exists: {os.path.exists(CODE_DIR)}")
print(f"Python path includes CODE_DIR: {CODE_DIR in sys.path}")