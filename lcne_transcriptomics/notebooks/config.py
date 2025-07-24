# config.py
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============= PATH CONFIGURATION =============
# Get project root directory (assuming config.py is in lcne_transcriptomics/notebooks/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define standard paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# Sub-directories for different data types
SNRNA_DATA_DIR = os.path.join(DATA_DIR, "snRNA")
MERFISH_DATA_DIR = os.path.join(DATA_DIR, "merfish")
RETROSEQ_DATA_DIR = os.path.join(DATA_DIR, "retroseq")
MESH_DIR = os.path.join(DATA_DIR, "mesh")

# Output directories for figures
SNRNA_FIGURE_DIR = os.path.join(FIGURE_DIR, "snRNA")
MERFISH_FIGURE_DIR = os.path.join(FIGURE_DIR, "merfish")
RETROSEQ_FIGURE_DIR = os.path.join(FIGURE_DIR, "retroseq")

# Create necessary directories
for directory in [DATA_DIR, OUTPUT_DIR, FIGURE_DIR, 
                 SNRNA_DATA_DIR, MERFISH_DATA_DIR, RETROSEQ_DATA_DIR, MESH_DIR,
                 SNRNA_FIGURE_DIR, MERFISH_FIGURE_DIR, RETROSEQ_FIGURE_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============= PLOT CONFIGURATION =============
# Default font path (modify as needed)
FONT_PATH = '/home/shuonan.chen/miniconda3/envs/allensdk/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/Helvetica.ttc'

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
def save_figure(filename, dir_path=FIGURE_DIR, formats=["svg", "png"], dpi=300):
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