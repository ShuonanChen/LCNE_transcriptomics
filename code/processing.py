import pandas as pd
import anndata
import numpy as np
from scipy.stats import rankdata
import utils


def rankrows(M, standardize = True):
    try:
        result = M.toarray().copy() # a x b
    except:
        result = M.copy()
    result = np.apply_along_axis(rankdata, 1, result) # if axis = 1, ranking for each row / cells        
    if standardize:
        means = np.mean(result, axis=1)
        stds = np.std(result, axis=1, ddof=0)
        stds[stds == 0] = 1e-10
        result = (result - means[:,None]) / stds[:,None]
    return result


def flip(a, xm):
        return(2*xm-a)

    
def get_hemi(S_mer, meshhome=None):
    '''
    assume the axis of interest are both on the last axis. 
    '''
    if meshhome !=None:
        allmeshes = utils.load_sym_mesh(meshhome)
        mesh = allmeshes[-1]
    xm = np.min(mesh.vertices[:,-1]) + np.ptp(mesh.vertices[:,-1])/2 # this is the center line to indicate the hemisphere 
    new_coords = S_mer.copy()
    new_coords[:,-1] = np.where(new_coords[:,-1] > xm, flip(new_coords[:,-1],xm), new_coords[:,-1])    
    return(new_coords)

def mirror_mesh_from_ref(mesh_to_mirror, meshhome):
    import trimesh
    allmeshes = utils.load_sym_mesh(meshhome)
    ref_mesh = allmeshes[-1]
    xm = np.min(ref_mesh.vertices[:, -1]) + np.ptp(ref_mesh.vertices[:, -1]) / 2
    verts = mesh_to_mirror.vertices.copy()
    verts[:, -1] = flip(verts[:, -1], xm)
    mirrored_mesh = trimesh.Trimesh(
        vertices=verts,
        faces=mesh_to_mirror.faces,
        process=False
    )
    return mirrored_mesh

def make_bilateral_mesh_from_ref(mesh_to_mirror, meshhome):
    ''' this is to put the mesh to both sides per request on figs
    '''
    import trimesh
    mirrored_mesh = mirror_mesh_from_ref(mesh_to_mirror, meshhome)
    mesh_both = trimesh.util.concatenate([mesh_to_mirror, mirrored_mesh])
    return mesh_both

def normalize_cols(M, ranked=True):
    """
    not use
    """
    raise RuntimeError("normalize_cols is deprecated - use `rankrows` instead")
    try:
        result = M.toarray().copy() # a x b
    except:
        result = M.copy()
    if ranked:  # output shape: 
        result = np.apply_along_axis(rankdata, 0, result) 
    means = np.mean(result, axis=0)
    stds = np.std(result, axis=0, ddof=0)
    stds[stds == 0] = 1e-10
    result = (result - means) / stds
    return result



def jitterspots(allspots, scl_jitter=0.05, rand_seed = 888):
    '''
    allspots are expected to be 3d spots 
    '''
    x = allspots[:,0]
    y = allspots[:,1]
    z = allspots[:,2]
    jitter_scale_x = scl_jitter * (x.max() - x.min())
    jitter_scale_y = scl_jitter * (y.max() - y.min())
    jitter_scale_z = scl_jitter * (z.max() - z.min())
    
    np.random.seed(rand_seed)
    x_jit = x + np.random.uniform(-jitter_scale_x, jitter_scale_x, size=len(x))
    y_jit = y + np.random.uniform(-jitter_scale_y, jitter_scale_y, size=len(y))
    z_jit = z + np.random.uniform(-jitter_scale_z, jitter_scale_z, size=len(z))

    return(np.c_[x_jit,y_jit,z_jit])