# config.py
import os, sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============= PATH CONFIGURATION =============
# Determine paths based on Code Ocean environment
# In reproducible runs: /code is working directory, /data and /results are mounted
if os.path.exists('/code'):
    # Code Ocean environment
    PROJECT_ROOT = '/code'
    DATA_DIR = '/data'
    OUTPUT_DIR = '/results'
else:
    # Local/IDE environment
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Add the code directory to Python path so modules can be imported
CODE_DIR = os.path.join(PROJECT_ROOT, "code") if not os.path.exists('/code') else PROJECT_ROOT
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# Define standard paths
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# Sub-directories for different data types
SNRNA_DATA_DIR = os.path.join(DATA_DIR, "snRNAseq_LCNE_batchcorrected")
MERFISH_DATA_DIR = os.path.join(DATA_DIR, "merfish")
RETROSEQ_DATA_DIR = os.path.join(DATA_DIR, "retroseq")
MESH_DIR = os.path.join(DATA_DIR, "LC_percentile_meshes")
MESH_DIR_sym = os.path.join(DATA_DIR, "mesh")
OTHERS_DIR = os.path.join(DATA_DIR, "others")
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
# Default font path (modify as needed)
FONT_PATH = os.path.join(PROJECT_ROOT, 'fonts', 'Helvetica.ttc')


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
def save_figure(filename, dir_path=FIGURE_DIR, formats=["svg", "png"], dpi=500):
    """
    Save figure in multiple formats
    
    Parameters:
    -----------
    filename : str
        Figure filename (without path or extension)
    dir_path : str, default=FIGURE_DIR
        Directory to save figure in
    formats : list, default=["svg", "png"]
        List of formats to save figure in
    dpi : int, default=300
        Resolution for raster formats
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

# ============= PROJECT CONSTANTS =============
# Add any project-specific constants here
RANDOM_SEED = 42


# Print debug information
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"CODE_DIR: {CODE_DIR}")
print(f"CODE_DIR exists: {os.path.exists(CODE_DIR)}")
print(f"Python path includes CODE_DIR: {CODE_DIR in sys.path}")