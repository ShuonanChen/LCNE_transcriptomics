import trimesh    
import numpy as np

def load_sym_mesh(meshhome):
    """
    Load three mesh objects for LC, CD and CV.
    meshhome
    """
    mesh_LC = trimesh.load_mesh(meshhome+"/LC_ccf_v1_250102 2.obj")
    mesh_CD = trimesh.load_mesh(meshhome+"/subCD_ccf_v1_250102 2.obj")
    mesh_CV = trimesh.load_mesh(meshhome+"/subCV_ccf_v1_250102 2.obj")
    allmeshes = [mesh_LC,mesh_CD,mesh_CV]
    return allmeshes



def load_mesh(meshhome,
              nameconstrains = '*67*'
             ):
    """
    Load all meshes (specify the string later)
    """

    import glob
    allmeshfiles = np.sort(glob.glob(meshhome+'/'+ nameconstrains))
    meshdict = dict()
    for f in allmeshfiles:
        meshdict[f.split('/')[-1].split('.')[0]] = trimesh.load_mesh(f)
    return meshdict



def ccf_pts_convert_to_mm(ccf_pts, bregma_points=None, ccf_res=None):
    '''
    copy'ed from sue's code ocean
    '''
    if bregma_points is None:
        bregma_points = np.array([216, 18, 228])
    if ccf_res is None:
        ccf_res = 25
    ccf_pts_mm = (ccf_pts - bregma_points) * ccf_res / 1000  # Convert to mm
    if np.size(ccf_pts_mm,0) == 1:
        ccf_pts_mm[0] = -1 * ccf_pts_mm[0]  # flip AP-axis
    else:
        ccf_pts_mm[:, 0] = -1 * ccf_pts_mm[:, 0]  # flip AP-axis
    return ccf_pts_mm



def gene_lengths_from_gtf(gtf_path):
    import pandas as pd 
    import gzip

    records = []

    open_func = gzip.open if gtf_path.endswith(".gz") else open
    with open_func(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if fields[2] != "exon":
                continue

            chrom, source, feature, start, end, score, strand, frame, attr = fields

            attr_dict = {
                x.split(" ")[0]: x.split(" ")[1].replace('"', '')
                for x in attr.strip(";").split("; ")
            }

            gene_id = attr_dict.get("gene_name", None)
            if gene_id is None:
                continue

            length = int(end) - int(start) + 1
            records.append((gene_id, length))

    df = pd.DataFrame(records, columns=["gene", "exon_len"])

    # Sum exon lengths per gene
    gene_lengths = df.groupby("gene")["exon_len"].sum()

    return gene_lengths  # pandas Series indexed by gene name



def tpm_normalize(X, gene_lengths_bp):
    import scipy.sparse as sp
    gene_lengths_kb = gene_lengths_bp / 1e3

    if sp.issparse(X):
        # Divide each gene by its length (column-wise)
        X_len = X.multiply(1.0 / gene_lengths_kb)
        
        # Per-cell scaling
        scale = X_len.sum(axis=1).A1
        scale[scale == 0] = 1e-10
        
        X_tpm = X_len.multiply(1e6 / scale[:, None])
        return X_tpm

    else:
        X_len = X / gene_lengths_kb
        scale = X_len.sum(axis=1, keepdims=True)
        scale[scale == 0] = 1e-10
        return X_len / scale * 1e6
    
    