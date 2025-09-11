# python core modules
import os

# third party modules
import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional
import matplotlib as mpl
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.cluster import hierarchy
import warnings
warnings.filterwarnings('ignore')


def isr_heatmap(adata, 
                     cluster_label='subleiden', 
                     isr_mtx=None,
                     rss_df=None,
                     topn=None,  # If None, show all regulons
                     save=False,
                     filename='isr_heatmap.pdf',
                     figsize=(12, 8),
                     row_cluster=True,
                     col_cluster=True,
                     cmap="YlGnBu",
                     vmin=None,
                     vmax=None,
                     yticklabels=True,
                     xticklabels=True,
                     show_cell_type_colors=True):
    """
    Create a comprehensive ISR (regulon activity) heatmap with proper cell type annotations
    
    Parameters:
    -----------
    adata : AnnData object
        Single cell data with regulon activity scores
    cluster_label : str
        Column name in adata.obs containing cell type/cluster annotations
    isr_mtx : pd.DataFrame or None
        ISR matrix (cells x regulons). If None, will use adata.obsm['isr']
    rss_df : pd.DataFrame or None
        RSS scores (cell_types x regulons). If None, will use adata.uns['rss']
    topn : int or None
        Number of top regulons per cell type to show. If None, shows all
    save : bool
        Whether to save the figure
    filename : str
        Filename for saving
    figsize : tuple
        Figure size
    row_cluster : bool
        Whether to cluster rows (cells)
    col_cluster : bool
        Whether to cluster columns (regulons)
    cmap : str
        Colormap name
    vmin, vmax : float
        Color scale limits
    yticklabels, xticklabels : bool
        Whether to show axis labels
    show_cell_type_colors : bool
        Whether to show cell type color annotations
    """
    
    # Get ISR matrix
    if isr_mtx is None:
        if 'isr' in adata.obsm:
            isr_mtx = pd.DataFrame(adata.obsm['isr'], 
                                  index=adata.obs_names, 
                                  columns=adata.var_names if adata.obsm['isr'].shape[1] == len(adata.var_names) else [f'Regulon_{i}' for i in range(adata.obsm['isr'].shape[1])])
        else:
            raise ValueError("ISR matrix not found. Please provide isr_mtx parameter or ensure adata.obsm['isr'] exists")
    
    # Get RSS scores if available and topn is specified
    selected_regulons = None
    if topn is not None and rss_df is not None:
        # Get top N regulons per cell type
        top_regulons = set()
        cell_types = adata.obs[cluster_label].unique()
        
        for ct in cell_types:
            if ct in rss_df.index:
                ct_top_regulons = rss_df.loc[ct].nlargest(topn).index
                top_regulons.update(ct_top_regulons)
        
        # Filter ISR matrix to only include top regulons
        available_regulons = set(isr_mtx.columns).intersection(top_regulons)
        if available_regulons:
            selected_regulons = list(available_regulons)
            isr_mtx = isr_mtx[selected_regulons]
        else:
            print(f"Warning: No overlap between top regulons and ISR matrix columns. Using all regulons.")
    
    # Get cell type annotations
    cell_types = adata.obs[cluster_label]
    
    # Create mean ISR per cell type for better visualization
    cell_type_means = []
    cell_type_names = []
    
    for ct in sorted(cell_types.unique()):
        ct_cells = cell_types == ct
        ct_mean = isr_mtx.loc[ct_cells].mean(axis=0)
        cell_type_means.append(ct_mean)
        cell_type_names.append(ct)
    
    # Create DataFrame with cell type means
    heatmap_data = pd.DataFrame(cell_type_means, 
                               index=cell_type_names,
                               columns=isr_mtx.columns)
    
    # Set up the figure
    plt.figure(figsize=figsize)
    
    # Create clustermap if clustering is requested
    if row_cluster or col_cluster:
        # Prepare row colors for cell types if requested
        row_colors = None
        if show_cell_type_colors:
            # Create color palette for cell types
            n_clusters = len(cell_type_names)
            colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
            cluster_colors = dict(zip(cell_type_names, colors))
            row_colors = pd.Series([cluster_colors[ct] for ct in cell_type_names], 
                                 index=cell_type_names, name='Cell Type')
        
        # Create clustermap
        g = sns.clustermap(heatmap_data,
                          row_cluster=row_cluster,
                          col_cluster=col_cluster,
                          cmap=cmap,
                          vmin=vmin,
                          vmax=vmax,
                          xticklabels=xticklabels,
                          yticklabels=yticklabels,
                          row_colors=row_colors if show_cell_type_colors else None,
                          figsize=figsize,
                          cbar_kws={'label': 'ISR Score'})
        
        # Improve axis labels
        g.ax_heatmap.set_xlabel('Regulons', fontsize=12)
        g.ax_heatmap.set_ylabel('Cell Types', fontsize=12)
        
        # Rotate x-axis labels for better readability
        if xticklabels:
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
        
        plt.sca(g.ax_heatmap)
        
    else:
        # Create simple heatmap without clustering
        plt.figure(figsize=figsize)
        sns.heatmap(heatmap_data,
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   xticklabels=xticklabels,
                   yticklabels=yticklabels,
                   cbar_kws={'label': 'ISR Score'})
        
        plt.xlabel('Regulons', fontsize=12)
        plt.ylabel('Cell Types', fontsize=12)
        
        # Rotate x-axis labels for better readability
        if xticklabels:
            plt.xticks(rotation=45, ha='right')
    
    plt.title(f'ISR Heatmap ({len(heatmap_data.columns)} regulons, {len(heatmap_data.index)} cell types)', 
              fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved as {filename}")
    
    plt.show()
    
    # Print summary information
    print(f"Heatmap created with:")
    print(f"  - {len(heatmap_data.index)} cell types: {', '.join(heatmap_data.index)}")
    print(f"  - {len(heatmap_data.columns)} regulons")
    print(f"  - Data range: {heatmap_data.values.min():.3f} to {heatmap_data.values.max():.3f}")
    
    return heatmap_data

def create_individual_cell_heatmap(adata,
                                  cluster_label='subleiden',
                                  isr_mtx=None,
                                  max_cells_per_type=50,
                                  selected_regulons=None,
                                  figsize=(15, 10),
                                  **kwargs):
    """
    Create heatmap showing individual cells (not averaged by cell type)
    
    Parameters:
    -----------
    max_cells_per_type : int
        Maximum number of cells to show per cell type (for visualization purposes)
    selected_regulons : list or None
        List of specific regulons to show. If None, shows all
    """
    
    # Get ISR matrix
    if isr_mtx is None:
        if 'isr' in adata.obsm:
            isr_mtx = pd.DataFrame(adata.obsm['isr'], 
                                  index=adata.obs_names)
        else:
            raise ValueError("ISR matrix not found")
    
    # Filter regulons if specified
    if selected_regulons is not None:
        available_regulons = set(isr_mtx.columns).intersection(selected_regulons)
        if available_regulons:
            isr_mtx = isr_mtx[list(available_regulons)]
    
    # Sample cells if too many
    cell_types = adata.obs[cluster_label]
    sampled_cells = []
    
    for ct in sorted(cell_types.unique()):
        ct_cells = adata.obs_names[cell_types == ct]
        if len(ct_cells) > max_cells_per_type:
            ct_cells = np.random.choice(ct_cells, max_cells_per_type, replace=False)
        sampled_cells.extend(ct_cells)
    
    # Filter ISR matrix to sampled cells
    plot_data = isr_mtx.loc[sampled_cells]
    
    # Create row colors for cell types
    sampled_cell_types = cell_types.loc[sampled_cells]
    unique_types = sorted(sampled_cell_types.unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    type_colors = dict(zip(unique_types, colors))
    row_colors = pd.Series([type_colors[ct] for ct in sampled_cell_types],
                          index=sampled_cells, name='Cell Type')
    
    # Create clustermap
    g = sns.clustermap(plot_data,
                      row_colors=row_colors,
                      figsize=figsize,
                      yticklabels=False,  # Too many cells to show labels
                      **kwargs)
    
    g.ax_heatmap.set_ylabel('Individual Cells', fontsize=12)
    g.ax_heatmap.set_xlabel('Regulons', fontsize=12)
    
    plt.title(f'Individual Cell ISR Heatmap ({len(sampled_cells)} cells, {len(plot_data.columns)} regulons)', 
              fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.show()
    
    return plot_data

# Example usage with your data:
"""
# For averaged cell type heatmap (recommended for interpretation)
heatmap_data = isr_heatmap(
    PDAC_subset,
    cluster_label='subleiden',
    isr_mtx=pd.DataFrame(PDAC_subset.obsm['isr'], 
                        index=PDAC_subset.obs_names,
                        columns=[f'Regulon_{i}' for i in range(PDAC_subset.obsm['isr'].shape[1])]),
    rss_df=PDAC_subset.uns['rss'] if 'rss' in PDAC_subset.uns else None,
    topn=10,  # Show top 10 regulons per cell type, or None for all
    figsize=(15, 8),
    cmap="YlGnBu",
    xticklabels=True,
    yticklabels=True
)

# For individual cell heatmap (if you want to see cell-level variation)
individual_data = create_individual_cell_heatmap(
    PDAC_subset,
    cluster_label='subleiden',
    max_cells_per_type=30, # Limit to 30 cells per type for visualization
    figsize=(20, 12),
    cmap="YlGnBu"
)
"""

def debug_isr_data(adata, cluster_label='subleiden'):
    """
    Debug function to understand your ISR data structure
    """
    print("=== ISR Data Debug Information ===")
    
    # Check ISR matrix
    if 'isr' in adata.obsm:
        isr_mtx = adata.obsm['isr']
        print(f"ISR matrix shape: {isr_mtx.shape}")
        print(f"ISR matrix type: {type(isr_mtx)}")
        
        if hasattr(isr_mtx, 'dtypes'):  # DataFrame
            print(f"Data types: {isr_mtx.dtypes.unique()}")
            print(f"Value range: {isr_mtx.min().min():.3f} to {isr_mtx.max().max():.3f}")
            print(f"NaN values: {isr_mtx.isna().sum().sum()}")
            print(f"Infinite values: {np.isinf(isr_mtx.values).sum()}")
            print(f"Zero values: {(isr_mtx == 0).sum().sum()}")
            print(f"Column names (first 5): {list(isr_mtx.columns[:5])}")
        else:  # numpy array
            print(f"Data type: {isr_mtx.dtype}")
            print(f"Value range: {isr_mtx.min():.3f} to {isr_mtx.max():.3f}")
            print(f"NaN values: {np.isnan(isr_mtx).sum()}")
            print(f"Infinite values: {np.isinf(isr_mtx).sum()}")
            print(f"Zero values: {(isr_mtx == 0).sum()}")
        print()
    
    # Check cell type annotations
    if cluster_label in adata.obs:
        cell_types = adata.obs[cluster_label]
        print(f"Cell types ({cluster_label}):")
        print(f"  - Number of cell types: {cell_types.nunique()}")
        print(f"  - Cell type names: {sorted(cell_types.unique())}")
        print(f"  - Cell type counts:")
        for ct, count in cell_types.value_counts().items():
            print(f"    {ct}: {count} cells")
        print()
    
    # Check RSS data if available
    if 'rss' in adata.uns:
        rss_data = adata.uns['rss']
        print(f"RSS data shape: {rss_data.shape}")
        print(f"RSS data type: {type(rss_data)}")
        if hasattr(rss_data, 'index'):
            print(f"RSS row names (cell types): {list(rss_data.index)}")
        if hasattr(rss_data, 'columns'):
            print(f"RSS column count (regulons): {len(rss_data.columns)}")
        print()
    
    return True

# Simple function to create a basic heatmap without clustering
def simple_isr_heatmap(adata, cluster_label='subleiden', figsize=(12, 6), min_variance=0.001, **kwargs):
    """
    Create a simple heatmap without clustering (more robust)
    """
    # Use the existing ISR matrix directly since it's already a DataFrame
    isr_mtx = adata.obsm['isr'].copy()
    
    print(f"Starting with ISR matrix shape: {isr_mtx.shape}")
    print(f"Regulon names: {list(isr_mtx.columns[:5])}...")
    
    # Clean data
    isr_mtx = isr_mtx.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Create cell type means
    cell_types = adata.obs[cluster_label]
    cell_type_means = []
    cell_type_names = []
    
    for ct in sorted(cell_types.unique()):
        ct_cells = cell_types == ct
        if ct_cells.sum() > 0:  # Make sure we have cells for this type
            ct_mean = isr_mtx.loc[ct_cells].mean(axis=0)
            cell_type_means.append(ct_mean)
            cell_type_names.append(ct)
            print(f"Added {ct}: {ct_cells.sum()} cells, mean range {ct_mean.min():.3f}-{ct_mean.max():.3f}")
    
    heatmap_data = pd.DataFrame(cell_type_means, 
                               index=cell_type_names,
                               columns=isr_mtx.columns)
    
    print(f"Heatmap data shape before filtering: {heatmap_data.shape}")
    print(f"Data range: {heatmap_data.min().min():.3f} to {heatmap_data.max().max():.3f}")
    
    # Instead of removing all-zero columns, remove columns with very low variance
    col_variance = heatmap_data.var(axis=0)
    print(f"Variance range: {col_variance.min():.6f} to {col_variance.max():.6f}")
    
    high_var_cols = col_variance > min_variance
    print(f"Keeping {high_var_cols.sum()} regulons with variance > {min_variance}")
    
    if high_var_cols.sum() == 0:
        print("Warning: No regulons with sufficient variance. Using all regulons.")
        filtered_data = heatmap_data
    else:
        filtered_data = heatmap_data.loc[:, high_var_cols]
    
    print(f"Final heatmap data shape: {filtered_data.shape}")
    
    if filtered_data.empty or filtered_data.shape[1] == 0:
        print("Error: No data left after filtering!")
        return None
    
    # Create simple heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(filtered_data, 
                cmap='YlOrRd',
                xticklabels=False if filtered_data.shape[1] > 20 else True,
                yticklabels=True,
                cbar_kws={'label': 'Mean ISR Score', 'pad': 0.15})
    
    plt.title(f'ISR Heatmap: {len(filtered_data.index)} cell types × {len(filtered_data.columns)} regulons')
    plt.xlabel('Regulons', fontsize=16)
    plt.ylabel('Cell Types', fontsize=16)
    plt.xticks(rotation=45, fontsize=16)  # Adjusted x-axis rotation and font size
    plt.yticks(fontsize=16)  # Adjusted y-axis font size
    plt.tight_layout()
    plt.show()
    
    return filtered_data



def plot_spatial_auc(
    adata,
    c,
    transcription_factor,
    dot_size=50,
    figure_size=(8, 6),
    spatial_layer='spatial',
    subset=False,
    subset_column=None,
    sample=None
):
    """
    Plot AUC scores for a transcription factor in spatial coordinates using Matplotlib.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing spatial coordinates and AUC matrices.
    c : pandas.DataFrame
        AUC matrix from .obsm['auc_mtx'] or .obsm['rep_auc_mtx'].
    transcription_factor : str
        Name of the transcription factor or regulon (e.g., 'JUN(+)' or 'JUN').
    dot_size : float, optional (default=50)
        Size of scatter plot points.
    figure_size : tuple, optional (default=(8, 6))
        Size of the figure (width, height).
    spatial_layer : str, optional (default='spatial')
        Key in adata.obsm for spatial coordinates.
    subset : bool, optional (default=False)
        If True, plot only a subset of cells based on subset_column and sample.
    subset_column : str, optional (default=None)
        Column in adata.obs to subset cells (e.g., 'patient').
    sample : str, optional (default=None)
        Value in subset_column to filter cells (e.g., 'W_C4').
    
    Returns:
    --------
    None
        Displays and saves the spatial plot with equal x/y aspect ratio, y-axis ticks on left,
        no x/y axis ticks, increased colorbar tick/label font size, larger title font size,
        title adjusted based on AUC matrix ('TF Regulon' for auc_mtx, 'TF Regulon Receptors' for rep_auc_mtx),
        white background, and a solid black frame.
    """
    # Verify inputs
    assert spatial_layer in adata.obsm, f"Spatial layer '{spatial_layer}' not in adata.obsm"
    assert transcription_factor in c.columns, f"Transcription factor '{transcription_factor}' not in {c.columns}"
    assert adata.obsm[spatial_layer].shape[1] >= 2, "Spatial layer must have at least 2 dimensions"
    
    # Get spatial coordinates and AUC values
    spatial_coords = adata.obsm[spatial_layer][:, :2]  # Take first 2 columns (x, y)
    auc_values = c[transcription_factor].values
    
    # Subset cells if requested
    if subset:
        assert subset_column in adata.obs.columns, f"Subset column '{subset_column}' not in adata.obs"
        assert sample is not None, "Sample must be provided when subset=True"
        mask = adata.obs[subset_column] == sample
        spatial_coords = spatial_coords[mask]
        auc_values = auc_values[mask]
        assert len(auc_values) > 0, f"No cells found for {subset_column} == {sample}"
    
    # Determine title based on AUC matrix
    if c is adata.obsm['auc_mtx']:
        title_prefix = f"{transcription_factor} Regulon"
    elif c is adata.obsm['rep_auc_mtx']:
        title_prefix = f"{transcription_factor} Regulon Receptors"
    else:
        title_prefix = f"{transcription_factor} Activity"
    
    # Create plot
    plt.figure(figsize=figure_size)
    ax = plt.gca()
    ax.set_facecolor('white')  # Set axes background to white
    plt.gcf().set_facecolor('white')  # Set figure background to white
    
    # Set solid black frame by enabling and styling spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    scatter = ax.scatter(
        spatial_coords[:, 0],
        spatial_coords[:, 1],
        c=auc_values,
        cmap='magma',
        s=dot_size
    )
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Move y-axis ticks to the left and remove all ticks
    ax.yaxis.tick_left()
    ax.set_xticks([])  # Remove x-axis ticks and labels
    ax.set_yticks([])  # Remove y-axis ticks and labels
    
    # Add colorbar with increased padding and larger tick/label font size
    cbar = plt.colorbar(scatter, label=f'{transcription_factor} AUC', pad=0.1)
    cbar.ax.tick_params(labelsize=14)  # Colorbar tick font size
    cbar.set_label(f'{transcription_factor} AUC', fontsize=16)  # Colorbar label font size
    
    # Set title and labels with larger title font size
    plt.title(title_prefix + (f' ({subset_column}: {sample})' if subset else ''), fontsize=20)
    #plt.xlabel('X', fontsize=12)
    #plt.ylabel('Y', fontsize=12)
    
    # Save plot
    plt.show()

# Example usage:
# plot_spatial_auc(PDAC_subset, PDAC_subset.obsm['auc_mtx'], 'JUN(+)', dot_size=50, figure_size=(8, 6))
# plot_spatial_auc(PDAC_subset, PDAC_subset.obsm['rep_auc_mtx'], 'SOX9', subset=True, subset_column='patient', sample='W_C4')