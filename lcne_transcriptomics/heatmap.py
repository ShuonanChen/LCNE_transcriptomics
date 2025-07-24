import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns



def downsample_moving_avg(X, n_out=100):
    """
    X: array of shape (sample x gene)
    n_out: desired number of output columns per row
    Returns: array of shape (n_rows, n_out)
    """
    n_rows, n_cols = X.shape

    # 1) choose window size w for the moving average
    #    here we floor(cin/n_out) to get roughly equal smoothing
    w = n_cols // n_out
    if w < 1:
        raise ValueError("n_out must be <= n_cols")

    # precompute the moving‐average filter kernel
    kernel = np.ones(w) / w

    # 2) for each row: convolve then resample
    Y = np.zeros((n_rows, n_out))
    for i in range(n_rows):
        row = X[i]
        # 'valid' mode gives length = n_cols - w + 1
        sm = np.convolve(row, kernel, mode='valid')

        # pick n_out indices evenly spaced across sm
        idx = np.linspace(0, sm.size - 1, n_out).round().astype(int)
        Y[i] = sm[idx]
    return Y


def scaling(hm, col_scale = True):
    '''
    we assume that the rows are genes and columns are the bins. so its col-wise scaling
    '''
    hm_out = hm.copy()
    minval = np.min(hm,1)
    hm_out -= minval[:,None]
    hm_out /= np.max(hm_out, 1)[:,None]
    return(hm_out) 
    
    

def reorder_genes(reduced):
    '''
    reduced: rows are gene, columns are binned samples. (G, n_bins)
    output (ordered_data): same size!
    '''
    Z = linkage(reduced, method='ward', metric='euclidean', optimal_ordering=True)
    dendro = dendrogram(Z, no_plot=True)
    row_order = dendro['leaves']
    ordered_data = reduced[row_order]
    return(ordered_data,row_order)



def get_final_heatmap(inputheatmap):
    '''
    inputheatmap: 
        should be G x N (but N needs to be ordered to a certain way. this will not change) 
        eg: adata[:,allyourgenes].X[trajectory['order']].copy()[np.argsort(cellscores)].T
        
    '''
    reduced = downsample_moving_avg(inputheatmap, n_out=50) # (G, n_out)
    reduced = scaling(reduced)
    ordered_data, row_order = reorder_genes(reduced)
    return(ordered_data, row_order)
    
    
    
    
    