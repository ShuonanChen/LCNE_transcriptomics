import trimesh    
import numpy as np

def load_sym_mesh(meshhome):
    """
    Load three mesh objects for LC, CD and CV.
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