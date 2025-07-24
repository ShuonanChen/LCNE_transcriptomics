from typing import Any, Tuple, List
from . import utils



def save_pdf(plt, outdir=None):
    if outdir is None:
        outdir = './foo.pdf'
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.gcf().savefig(outdir)
    
    
def format_ticklabels(ticks, scale=25):
        return [f'{int(t * scale)}' if i % 2 == 0 else '' for i, t in enumerate(ticks)]


def draw_scale_bar(ax, length_px, px_to_mm=25*1e-3, loc=(0.1, 0.05), linewidth=4):
    """Draws a horizontal scale bar of length_px on ax."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + loc[0] * (xlim[1] - xlim[0])
    y0 = ylim[0] + loc[1] * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + length_px],[y0, y0],color='black',linewidth=linewidth)
    ax.text(x0 + length_px/2,  y0 - 3,#0.03*(ylim[1]-ylim[0]),
            f'{length_px * px_to_mm:.1f} mm',ha='center', va='bottom')

    
def plot_mesh(ax: Any, allmeshes, 
              direction: str = 'c', 
             meshcol = 'lightgray') -> None:
    """
    Plot the three meshes on the given axis.
    parameter direction: select index to choose coordinate ('c' uses index 2, otherwise index 0)
    allmeshes is a dictionary now
    """
    import trimesh
    ax.set_aspect('equal')
    i = 2 if direction == 'c' else 0
    if isinstance(allmeshes, dict):  #, trimesh.Trimesh
        for k,mesh in allmeshes.items():
            ax.triplot(mesh.vertices.T[i], mesh.vertices.T[1], mesh.faces, alpha=0.4, label=k, color =meshcol)
    elif isinstance(allmeshes, trimesh.Trimesh):
        ax.triplot(allmeshes.vertices.T[i], allmeshes.vertices.T[1], allmeshes.faces, alpha=0.4, color =meshcol)
    else:
        print('wrong mesh input') 
    
    ax.invert_yaxis()
    
from sklearn.preprocessing import LabelEncoder

def plot_spatial(ax,
                 *,
                 coords=None,
                 colorvalues=None,
                 adata=None,
                 color=None,
                 basis='spatial',
                 dims=(2,1),
                 s=10,
                 cmap='viridis',
#                  vmin=0, vmax=1,
                 alpha=0.8,
                 meshes=None,
                 xlabel='',
                 ylabel='',
                 scale_px=40,
                 scale_color='black',
                 scale_linewidth=4,
                 changeticks=False,
                 colorbartitle=None
                ):
    
    """
    ax : matplotlib Axes
    coords : (N,3) array
    colorvalues : length-N array for coloring -> example: imputed_values
    meshes : dict of name->(vertices, faces)
    dims : which two dims of coords to plot (x_dim, y_dim)

    If `adata` is given, plots using generic matplotlib (instead of scanpy).
    Otherwise, does a raw scatter of `coords` + `colorvalues`.
    In both cases, overlays meshes + draws a scale bar + fixes aspect.
    
    example usage with adata:
    example usage with coors:
    """
    if meshes is None:
        meshes = utils.load_mesh()
    mesh_direction='c' if dims[0]==2 else 's'
    
    xdim, ydim = dims

    if adata is not None:
        # Handle categorical color values
        if color is not None:
            # Convert categorical values to numeric
            le = LabelEncoder()
            colorvalues = le.fit_transform(adata.obs[color].values)

        ax.scatter(adata.obsm[basis][:, xdim],
                   adata.obsm[basis][:, ydim],
                   s=s,
                   alpha=alpha,
                   c=colorvalues,
                   cmap=cmap,
                  )
    else:
        # When `adata` is not provided, fallback to using the coords
        sca = ax.scatter(coords[:, dims[0]],
                         coords[:, dims[1]],
                         s=s,
                         alpha=alpha,
                         c=colorvalues,
                         cmap=cmap,
#                          vmin=vmin,
#                          vmax=vmax
                        )
        cbar = ax.figure.colorbar(sca, ax=ax)
        if colorbartitle is not None:
            cbar.set_label(colorbartitle, loc='center')
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # overlay meshes
    plot_mesh(ax, meshes, direction = mesh_direction)
    
    if changeticks:
        xt = ax.get_xticks(); yt = ax.get_yticks()
        ax.set_xticks(xt); ax.set_yticks(yt)
        ax.set_xticklabels(format_ticklabels(xt))
        ax.set_yticklabels(format_ticklabels(yt))

    
    # scale bar
    draw_scale_bar(ax,length_px=scale_px,linewidth=scale_linewidth)
    leg = ax.get_legend()
    if leg:
        leg.remove()
