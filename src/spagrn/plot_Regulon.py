# python core modules
import os
import uuid
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
from scipy.stats import mannwhitneyu, ttest_ind, kruskal, f_oneway
from itertools import combinations
import statsmodels.stats.multitest as smm
from collections import namedtuple
import warnings
warnings.filterwarnings('ignore')








def isr_heatmap(adata, 
                cluster_label,  # REMOVED DEFAULT
                isr_mtx=None,
                rss_df=None,
                topn=None,
                selected_regulons=None,
                excluded_cell_types=None,
                included_cell_types=None,
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
                ytick_rotation=0,
                scale=False):
    """
    Create a comprehensive ISR (regulon activity) heatmap with proper cell type annotations
    
    Parameters:
    -----------
    adata : AnnData object
        Single cell data with regulon activity scores
    cluster_label : str
        Column name in adata.obs containing cell type/cluster annotations (NO DEFAULT)
    isr_mtx : pd.DataFrame or None
        ISR matrix (cells x regulons). If None, will use adata.obsm['isr']
    rss_df : pd.DataFrame or None
        RSS scores (cell_types x regulons). If None, will use adata.uns['rss']
    topn : int or None
        Number of top regulons per cell type to show. If None, shows all unless selected_regulons is specified
    selected_regulons : list or None
        List of specific regulon names to show. If not None, overrides topn and shows only these regulons
    excluded_cell_types : list or None
        List of cell type names to exclude from the plot. If None, includes all cell types.
    included_cell_types : list or None
        List of cell type names to include in the plot. If None, includes all cell types.
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
    ytick_rotation : float
        Rotation angle for y-axis tick labels (default: 0)
    scale : bool
        Whether to scale values by column (regulon) using z-score normalization (default: False)
    """
    
    # Get ISR matrix
    if isr_mtx is None:
        if 'isr' in adata.obsm:
            isr_mtx = pd.DataFrame(adata.obsm['isr'], 
                                  index=adata.obs_names, 
                                  columns=adata.var_names if adata.obsm['isr'].shape[1] == len(adata.var_names) else [f'Regulon_{i}' for i in range(adata.obsm['isr'].shape[1])])
        else:
            raise ValueError("ISR matrix not found. Please provide isr_mtx parameter or ensure adata.obsm['isr'] exists")
    
    # Get cell type annotations
    cell_types = adata.obs[cluster_label].copy()
    
    # Cell types: use either included OR excluded, not both
    if included_cell_types is not None and excluded_cell_types is not None:
        raise ValueError("Cannot specify both included_cell_types and excluded_cell_types. Use one or the other.")
    
    # Filter by included_cell_types OR excluded_cell_types
    if included_cell_types is not None:
        mask = cell_types.isin(included_cell_types)
        cell_types = cell_types[mask]
        isr_mtx = isr_mtx.loc[mask]
        print(f"Included cell types: {included_cell_types}")
    elif excluded_cell_types is not None:
        mask = ~cell_types.isin(excluded_cell_types)
        cell_types = cell_types[mask]
        isr_mtx = isr_mtx.loc[mask]
        print(f"Excluded cell types: {excluded_cell_types}")
    
    print(f"Remaining cell types: {sorted(cell_types.unique())}")
    
    # Filter by selected regulons if specified
    if selected_regulons is not None:
        available_regulons = set(isr_mtx.columns).intersection(selected_regulons)
        if not available_regulons:
            raise ValueError(f"None of the selected regulons {selected_regulons} found in ISR matrix columns: {list(isr_mtx.columns)}")
        isr_mtx = isr_mtx[list(available_regulons)]
    # Otherwise, get RSS scores if available and topn is specified
    elif topn is not None and rss_df is not None:
        # Get top N regulons per cell type
        top_regulons = set()
        for ct in cell_types.unique():
            if ct in rss_df.index:
                ct_top_regulons = rss_df.loc[ct].nlargest(topn).index
                top_regulons.update(ct_top_regulons)
        
        # Filter ISR matrix to only include top regulons
        available_regulons = set(isr_mtx.columns).intersection(top_regulons)
        if available_regulons:
            isr_mtx = isr_mtx[list(available_regulons)]
        else:
            print(f"Warning: No overlap between top regulons and ISR matrix columns. Using all regulons.")
    
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
    
    # Apply column-wise scaling if requested
    if scale:
        from scipy.stats import zscore
        heatmap_data = heatmap_data.apply(zscore, axis=0)
        print(f"Applied z-score scaling by column (regulon)")
        print(f"  - Scaled data range: {heatmap_data.values.min():.3f} to {heatmap_data.values.max():.3f}")
    
    # Set up the figure
    plt.figure(figsize=figsize)
    
    # Determine if using clustermap (which has y labels on right)
    is_clustermap = row_cluster or col_cluster
    
    # Compute ha for y ticks based on side
    if is_clustermap:
        # y labels on right
        ha_y = 'left' if ytick_rotation == 0 else 'center'
    else:
        # y labels on left
        ha_y = 'right' if ytick_rotation == 0 else 'center'
    
    # Create clustermap if clustering is requested
    if is_clustermap:
        # Create clustermap without row colors
        g = sns.clustermap(heatmap_data,
                          row_cluster=row_cluster,
                          col_cluster=col_cluster,
                          cmap=cmap,
                          vmin=vmin,
                          vmax=vmax,
                          xticklabels=xticklabels,
                          yticklabels=yticklabels,
                          row_colors=None,
                          figsize=figsize,
                          dendrogram_ratio=(0.05, 0.1),
                          cbar_pos=(0.02, 0.8, 0.01, 0.01))
        
        # Add colorbar label
        cbar_label = 'ISR Score (z-score)' if scale else 'ISR Score'
        g.ax_cbar.set_ylabel(cbar_label, fontsize=14, rotation=90, labelpad=15)
        g.ax_cbar.tick_params(labelsize=11)
        
        # Improve axis labels
        g.ax_heatmap.set_xlabel('Regulons', fontsize=18)
        g.ax_heatmap.set_ylabel('Cell Types', fontsize=18)
        
        # Rotate x-axis labels and increase tick size for better readability
        if xticklabels:
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', va='top', fontsize=18)
        if yticklabels:
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=ytick_rotation, ha=ha_y, va='center', fontsize=18)
        
        # Move title above the top dendrogram
        title_text = f'ISR Heatmap ({len(heatmap_data.columns)} regulons, {len(heatmap_data.index)} cell types)'
        g.fig.suptitle(title_text, fontsize=18, y=0.98)
        
        plt.sca(g.ax_heatmap)
        
    else:
        # Create simple heatmap without clustering
        plt.figure(figsize=figsize)
        cbar_label = 'ISR Score (z-score)' if scale else 'ISR Score'
        ax = sns.heatmap(heatmap_data,
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   xticklabels=xticklabels,
                   yticklabels=yticklabels,
                   cbar_kws={'label': cbar_label, 'fraction': 0.02, 'aspect': 30})
        
        plt.xlabel('Regulons', fontsize=18)
        plt.ylabel('Cell Types', fontsize=18)
        
        # Rotate x-axis labels and increase tick size for better readability
        if xticklabels:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', va='top', fontsize=18)
        if yticklabels:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation, ha=ha_y, va='center', fontsize=18)
        
        plt.title(f'ISR Heatmap ({len(heatmap_data.columns)} regulons, {len(heatmap_data.index)} cell types)', 
                  fontsize=18, pad=20)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved as {filename}")
    
    plt.show()
    
    # Print summary information
    print(f"Heatmap created with:")
    print(f"  - {len(heatmap_data.index)} cell types: {', '.join(heatmap_data.index)}")
    print(f"  - {len(heatmap_data.columns)} regulons: {', '.join(heatmap_data.columns)}")
    print(f"  - Data range: {heatmap_data.values.min():.3f} to {heatmap_data.values.max():.3f}")
    
    return heatmap_data



def isr_violin(adata, 
               cell_type_label,
               selected_regulons=None,
               condition_label=None,
               excluded_cell_types=None,
               excluded_conditions=None,
               included_cell_types=None,
               included_conditions=None,
               save=False,
               filename='isr_violin.pdf',
               figsize=(12, 6),
               split=True,
               inner='box',
               palette=None,
               col_wrap=4,
               height=6,
               aspect=1,
               make_stack=True,
               xticklabel_size=10,
               xticklabel_rotation=45):
    """
    Create violin plots for ISR (regulon activity) scores across cell types and conditions.
    
    Parameters:
    -----------
    adata : AnnData object
        Single cell data with regulon activity scores in adata.obsm['isr']
    cell_type_label : str
        Column name in adata.obs containing cell type annotations (x-axis)
    selected_regulons : list, str, or None
        List of specific regulon names to plot, or a single string. If None, plots all regulons.
    condition_label : str or None
        Column name in adata.obs containing condition annotations (hue). If None, no hue is used.
    excluded_cell_types : list or None
        List of cell type names to exclude from the plot
    excluded_conditions : list or None
        List of condition names to exclude from the plot
    included_cell_types : list or None
        List of cell type names to include in the plot
    included_conditions : list or None
        List of condition names to include in the plot
    save : bool
        Whether to save the figure
    filename : str
        Filename for saving
    figsize : tuple
        Figure size (used to calculate aspect and height if single regulon)
    split : bool
        If True and hue has two categories, split the violin
    inner : str or None
        Style of the inner plot
    palette : dict or list or None
        Colors for the hue categories
    col_wrap : int
        Wrap facets after this many columns
    height : float
        Height of each facet in inches
    aspect : float
        Aspect ratio of each facet (width/height)
    make_stack : bool
        If True, create stacked violin plots with regulons on y-axis and cell types on x-axis (default: False)
    xticklabel_size : int
        Font size for x-axis tick labels (default: 10)
    xticklabel_rotation : int
        Rotation angle for x-axis tick labels (default: 45)
    """
    
    # Get ISR matrix from adata.obsm['isr']
    if 'isr' not in adata.obsm:
        raise ValueError("ISR matrix not found. Please ensure adata.obsm['isr'] exists")
    
    # Check if it's already a DataFrame with columns, if not convert it
    if isinstance(adata.obsm['isr'], pd.DataFrame):
        isr_mtx = adata.obsm['isr'].copy()
    else:
        # Convert to DataFrame, preserving index
        isr_mtx = pd.DataFrame(adata.obsm['isr'], index=adata.obs_names)
    
    # Handle selected regulons
    if selected_regulons is None:
        selected_regulons = list(isr_mtx.columns)
    elif isinstance(selected_regulons, str):
        selected_regulons = [selected_regulons]
    
    # Filter to selected regulons
    available_regulons = set(isr_mtx.columns).intersection(selected_regulons)
    if not available_regulons:
        raise ValueError(f"None of the selected regulons {selected_regulons} found in ISR matrix columns: {list(isr_mtx.columns)}")
    isr_mtx = isr_mtx[list(available_regulons)]
    
    # Get relevant obs columns
    obs_columns = [cell_type_label]
    if condition_label is not None:
        obs_columns.append(condition_label)
    obs_df = adata.obs[obs_columns].copy()
    
    # Create mask for inclusions or exclusions (mutually exclusive)
    mask = pd.Series(True, index=obs_df.index)
    
    # Cell types: use either included OR excluded, not both
    if included_cell_types is not None and excluded_cell_types is not None:
        raise ValueError("Cannot specify both included_cell_types and excluded_cell_types. Use one or the other.")
    
    if included_cell_types is not None:
        mask &= obs_df[cell_type_label].isin(included_cell_types)
        print(f"Included cell types: {included_cell_types}")
    elif excluded_cell_types is not None:
        mask &= ~obs_df[cell_type_label].isin(excluded_cell_types)
        print(f"Excluded cell types: {excluded_cell_types}")
    
    # Conditions: use either included OR excluded, not both
    if condition_label is not None:
        if included_conditions is not None and excluded_conditions is not None:
            raise ValueError("Cannot specify both included_conditions and excluded_conditions. Use one or the other.")
        
        if included_conditions is not None:
            mask &= obs_df[condition_label].isin(included_conditions)
            print(f"Included conditions: {included_conditions}")
        elif excluded_conditions is not None:
            mask &= ~obs_df[condition_label].isin(excluded_conditions)
            print(f"Excluded conditions: {excluded_conditions}")
    
    # Apply mask
    obs_df = obs_df.loc[mask]
    isr_mtx = isr_mtx.loc[mask]
    print(f"Remaining cell types: {sorted(obs_df[cell_type_label].unique())}")
    if condition_label is not None:
        print(f"Remaining conditions: {sorted(obs_df[condition_label].unique())}")
    
    # Create long-form DataFrame for plotting
    long_df = pd.melt(isr_mtx.reset_index(), 
                      id_vars=['index'], 
                      value_vars=list(isr_mtx.columns), 
                      var_name='Regulon', 
                      value_name='ISR Score')
    long_df = long_df.set_index('index')
    long_df = long_df.join(obs_df)
    
    # Remove unused categories from categorical columns to ensure only filtered data is plotted
    if pd.api.types.is_categorical_dtype(long_df[cell_type_label]):
        long_df[cell_type_label] = long_df[cell_type_label].cat.remove_unused_categories()
    if condition_label is not None and pd.api.types.is_categorical_dtype(long_df[condition_label]):
        long_df[condition_label] = long_df[condition_label].cat.remove_unused_categories()
    
    # Check if split is valid (only works with exactly 2 hue levels)
    use_split = False
    if condition_label is not None:
        n_conditions = obs_df[condition_label].nunique()
        use_split = split and n_conditions == 2
        if split and n_conditions != 2:
            print(f"Warning: split=True requires exactly 2 conditions, but found {n_conditions}. Setting split=False.")
    
    # Calculate height and aspect from figsize if only one regulon
    if len(selected_regulons) == 1:
        height = figsize[1]
        aspect = figsize[0] / figsize[1]
    
    # Create stacked violin plot if requested
    if make_stack:
        fig, axes = plt.subplots(len(selected_regulons), 1, figsize=figsize, sharex=True)
        if len(selected_regulons) == 1:
            axes = [axes]
        
        for idx, regulon in enumerate(selected_regulons):
            regulon_data = long_df[long_df['Regulon'] == regulon]
            
            sns.violinplot(
                data=regulon_data,
                x=cell_type_label,
                y='ISR Score',
                hue=condition_label if condition_label is not None else None,
                split=use_split,
                inner=inner,
                palette=palette,
                ax=axes[idx]
            )
            
            axes[idx].set_ylabel(f'{regulon}', fontsize=12)
            axes[idx].set_xlabel('')
            
            # Only show legend on first subplot
            if idx > 0 and condition_label is not None:
                axes[idx].get_legend().remove()
            
            # Rotate x-axis labels only on bottom plot
            if idx < len(selected_regulons) - 1:
                axes[idx].set_xticklabels([])
            else:
                axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=xticklabel_rotation, ha='right', fontsize=xticklabel_size)
        
        # Add x-axis label to bottom plot
        axes[-1].set_xlabel(cell_type_label, fontsize=12)
        
        # Add overall title
        fig.suptitle(f'ISR Regulon Score ({len(selected_regulons)} regulons)', fontsize=16, y=0.995)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Stacked violin plot saved as {filename}")
        
        # Don't call plt.show() here - return the figure instead
        
        # Print summary information
        print(f"Stacked violin plot created with:")
        print(f"  - Regulons: {', '.join(selected_regulons)}")
        print(f"  - Cell types: {', '.join(sorted(obs_df[cell_type_label].unique()))}")
        if condition_label is not None:
            print(f"  - Conditions: {', '.join(sorted(obs_df[condition_label].unique()))}")
        print(f"  - Number of cells: {len(long_df) // len(selected_regulons)}")
        
        return
        
    
    # Create the violin plot using catplot for faceting by regulon (original behavior)
    g = sns.catplot(
        data=long_df,
        x=cell_type_label,
        y='ISR Score',
        hue=condition_label if condition_label is not None else None,
        col='Regulon' if len(selected_regulons) > 1 else None,
        col_wrap=col_wrap if len(selected_regulons) > 1 else None,
        kind='violin',
        split=use_split,
        inner=inner,
        palette=palette,
        height=height,
        aspect=aspect,
        legend_out=True
    )
    
    # Set y-axis label based on number of regulons
    if len(selected_regulons) == 1:
        # For single regulon, use regulon name as y-axis label
        for ax in g.axes.flat:
            ax.set_ylabel(f'{selected_regulons[0]} ISR Score', fontsize=12)
    else:
        # For multiple regulons, keep generic label
        for ax in g.axes.flat:
            ax.set_ylabel('ISR Score', fontsize=12)
    
    # Rotate x-axis labels for readability
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xticklabel_rotation, ha='right', fontsize=xticklabel_size)
    
    # Add title
    g.fig.suptitle(f'ISR Violin Plots ({len(selected_regulons)} regulons)', y=1.02, fontsize=16)
    
    # Adjust layout
    g.tight_layout()
    
    if save:
        g.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Violin plot saved as {filename}")
    
    # Don't call plt.show() here either - catplot already displays
    
    # Print summary information
    print(f"Violin plot created with:")
    print(f"  - Regulons: {', '.join(selected_regulons)}")
    print(f"  - Cell types: {', '.join(sorted(obs_df[cell_type_label].unique()))}")
    if condition_label is not None:
        print(f"  - Conditions: {', '.join(sorted(obs_df[condition_label].unique()))}")
    print(f"  - Number of cells: {len(long_df) // len(selected_regulons)}")


def isr_ridge(adata, 
              cell_type_label,
              selected_regulons=None,
              excluded_cell_types=None,
              included_cell_types=None,
              save=False,
              filename='isr_ridge.pdf',
              figsize=(12, 8),
              palette=None,
              overlap=0.6,
              alpha=0.7,
              linewidth=1.5,
              fill=True,
              ncols=1,
              kde_kws=None,
              legend=True,
              style='stacked'):  
    """
    Create ridge plots for ISR (regulon activity) scores across cell types.
        
    Parameters:
    -----------
    style : str
        Ridge plot style. Options: 'stacked' or 'overlapping'
        - 'stacked': Traditional ridge plot with each cell type on a separate row
        - 'overlapping': All cell types overlaid on the same axis (LIKE YOUR REFERENCE IMAGE!)
        Default: 'stacked'
        
    Examples:
    ---------
    # Overlapping style (like your reference image)
    isr_ridge(adata, cell_type_label='niche4', selected_regulons='MAF(+)',
              style='overlapping', alpha=0.6,
              included_cell_types=['Plasma Niche (IgG)', 'Plasma Niche (IgA/M)'])
    """
    from scipy import stats
    
    # [All the data preparation code stays the same - omitted for brevity]
    # Get ISR matrix, filter regulons, apply masks, etc.
    
    if 'isr' not in adata.obsm:
        raise ValueError("ISR matrix not found. Please ensure adata.obsm['isr'] exists")
    
    if isinstance(adata.obsm['isr'], pd.DataFrame):
        isr_mtx = adata.obsm['isr'].copy()
    else:
        isr_mtx = pd.DataFrame(adata.obsm['isr'], index=adata.obs_names)
    
    if selected_regulons is None:
        selected_regulons = list(isr_mtx.columns)
    elif isinstance(selected_regulons, str):
        selected_regulons = [selected_regulons]
    
    available_regulons = set(isr_mtx.columns).intersection(selected_regulons)
    if not available_regulons:
        raise ValueError(f"None of the selected regulons found")
    isr_mtx = isr_mtx[list(available_regulons)]
    
    obs_columns = [cell_type_label]
    obs_df = adata.obs[obs_columns].copy()
    
    mask = pd.Series(True, index=obs_df.index)
    
    if included_cell_types is not None:
        mask &= obs_df[cell_type_label].isin(included_cell_types)
    elif excluded_cell_types is not None:
        mask &= ~obs_df[cell_type_label].isin(excluded_cell_types)
    
    obs_df = obs_df.loc[mask]
    isr_mtx = isr_mtx.loc[mask]
    
    long_df = pd.melt(isr_mtx.reset_index(), 
                      id_vars=['index'], 
                      value_vars=list(isr_mtx.columns), 
                      var_name='Regulon', 
                      value_name='ISR Score')
    long_df = long_df.set_index('index')
    long_df = long_df.join(obs_df)
    
    cell_types = sorted(obs_df[cell_type_label].unique())
    
    if palette is None:
        n_cell_types = len(cell_types)
        cmap = plt.cm.tab20 if n_cell_types <= 20 else plt.cm.tab20c
        colors = [cmap(i % 20) for i in range(n_cell_types)]
        palette = dict(zip(cell_types, colors))
    elif isinstance(palette, list):
        palette = dict(zip(cell_types, palette))
    
    n_regulons = len(selected_regulons)
    nrows = int(np.ceil(n_regulons / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_regulons == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each regulon
    for reg_idx, regulon in enumerate(selected_regulons):
        regulon_data = long_df[long_df['Regulon'] == regulon]
        ax = axes[reg_idx]
        
        # CALL THE HELPER FUNCTION WITH style PARAMETER
        _plot_single_ridge(ax, regulon_data, cell_types, cell_type_label,
                          palette, overlap, alpha, linewidth, fill, kde_kws, style)
        
        ax.set_ylabel(regulon, fontsize=12, fontweight='bold')
        if reg_idx == n_regulons - 1:
            ax.set_xlabel('ISR Score', fontsize=12)
    
    for idx in range(n_regulons, len(axes)):
        axes[idx].axis('off')
    
    if legend and style == 'overlapping':
        handles = [plt.Rectangle((0, 0), 1, 1, fc=palette[ct], alpha=alpha) 
                  for ct in cell_types]
        fig.legend(handles, cell_types, loc='center left', 
                  bbox_to_anchor=(1.0, 0.5), fontsize=10)
    
    title = f'ISR Ridge Plot - {style.capitalize()} Style'
    if len(selected_regulons) == 1:
        title += f' - {selected_regulons[0]}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def _plot_single_ridge(ax, data, cell_types, cell_type_label, palette, 
                       overlap, alpha, linewidth, fill, kde_kws, style='stacked'):
    """
    Helper function - THE KEY CHANGE IS HERE!
    Now supports both 'stacked' and 'overlapping' styles.
    """
    from scipy import stats
    
    if kde_kws is None:
        kde_kws = {}
    
    all_scores = data['ISR Score'].values
    x_min, x_max = all_scores.min(), all_scores.max()
    x_range = x_max - x_min
    x_padding = x_range * 0.1
    x_min -= x_padding
    x_max += x_padding
    
    x_vals = np.linspace(x_min, x_max, 500)
    
    if style == 'overlapping':
        # ========================================
        # OVERLAPPING STYLE - THIS IS WHAT YOU WANT!
        # All curves on same baseline, different colors
        # ========================================
        max_density = 0
        densities = []
        
        for cell_type in cell_types:
            ct_data = data[data[cell_type_label] == cell_type]['ISR Score'].values
            
            if len(ct_data) > 1:
                kde = stats.gaussian_kde(ct_data, **kde_kws)
                density = kde(x_vals)
                densities.append(density)
                max_density = max(max_density, density.max())
            else:
                densities.append(np.zeros_like(x_vals))
        
        # Plot each cell type on SAME baseline
        for cell_type, density in zip(cell_types, densities):
            color = palette[cell_type]
            
            if fill:
                ax.fill_between(x_vals, 0, density, 
                               color=color, alpha=alpha, linewidth=0, label=cell_type)
            
            ax.plot(x_vals, density, color=color, linewidth=linewidth, alpha=1.0)
        
        # Clean axes
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, max_density * 1.1)
        ax.set_ylabel('Density', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
        
    else:
        # ========================================
        # STACKED STYLE - Original behavior
        # Each cell type on different row
        # ========================================
        n_cell_types = len(cell_types)
        y_spacing = 1.0 * (1.0 - overlap)
        
        max_density = 0
        densities = []
        
        for cell_type in cell_types:
            ct_data = data[data[cell_type_label] == cell_type]['ISR Score'].values
            
            if len(ct_data) > 1:
                kde = stats.gaussian_kde(ct_data, **kde_kws)
                density = kde(x_vals)
                densities.append(density)
                max_density = max(max_density, density.max())
            else:
                densities.append(np.zeros_like(x_vals))
        
        max_height = y_spacing * 0.9
        scale_factor = max_height / max_density if max_density > 0 else 1
        
        for idx, (cell_type, density) in enumerate(zip(cell_types, densities)):
            y_baseline = idx * y_spacing
            scaled_density = density * scale_factor
            y_vals = y_baseline + scaled_density
            
            color = palette[cell_type]
            
            if fill:
                ax.fill_between(x_vals, y_baseline, y_vals, 
                               color=color, alpha=alpha, linewidth=0)
            
            ax.plot(x_vals, y_vals, color=color, linewidth=linewidth, alpha=1.0)
            
            ax.text(x_min - x_range * 0.02, y_baseline + y_spacing * 0.4, 
                   cell_type, va='center', ha='right', fontsize=10)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-y_spacing * 0.2, (n_cell_types - 1) * y_spacing + y_spacing * 1.2)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)


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
    
    g.ax_heatmap.set_ylabel('Individual Cells', fontsize=18)
    g.ax_heatmap.set_xlabel('Regulons', fontsize=18)
    
    plt.title(f'Individual Cell ISR Heatmap ({len(sampled_cells)} cells, {len(plot_data.columns)} regulons)', 
              fontsize=18, pad=20)
    
    plt.tight_layout()
    plt.show()
    
    return plot_data

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
    plt.xlabel('Regulons', fontsize=18)
    plt.ylabel('Cell Types', fontsize=18)
    plt.xticks(rotation=45, fontsize=18)  # Adjusted x-axis rotation and font size
    plt.yticks(fontsize=18)  # Adjusted y-axis font size
    plt.tight_layout()
    plt.show()
    
    return filtered_data







def isr_test(
    adata,
    cell_type_col,
    selected_regulons=None,
    selected_cell_types=None,
    excluded_cell_types=None,
    isr_matrix_key='isr',
    plot_style='violin',
    jitter=True,
    test='auto',
    post_hoc=True,
    show_asterisk=True,
    line_asterisk=True,
    asterisk_size=18,
    plot_dir=None,
    figsize=(14, 12),
    palette=None,
    split=False,
    inner='box',
    alpha=0.7,
    fontsize_xlabel=24,
    fontsize_ylabel=24,
    fontsize_xtick=24,
    fontsize_ytick=24,
    fontsize_title=24,
    rotation_xtick=45,
    show_stats_table=True,
    max_comparisons_to_show=10
):
    """
    Create a comprehensive ISR (regulon activity) heatmap with proper cell type annotations
    
    Parameters:
    -----------
    adata : AnnData object
        Single cell data with regulon activity scores
    cluster_label : str
        Column name in adata.obs containing cell type/cluster annotations (NO DEFAULT)
    isr_mtx : pd.DataFrame or None
        ISR matrix (cells x regulons). If None, will use adata.obsm['isr']
    rss_df : pd.DataFrame or None
        RSS scores (cell_types x regulons). If None, will use adata.uns['rss']
    topn : int or None
        Number of top regulons per cell type to show. If None, shows all unless selected_regulons is specified
    selected_regulons : list or None
        List of specific regulon names to show. If not None, overrides topn and shows only these regulons
    excluded_cell_types : list or None
        List of cell type names to exclude from the plot. If None, includes all cell types.
    included_cell_types : list or None
        List of cell type names to include in the plot. If None, includes all cell types.
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
    ytick_rotation : float
        Rotation angle for y-axis tick labels (default: 0)
    scale : bool
        Whether to scale values by column (regulon) using z-score normalization (default: False)
    """
    # ======================================== INPUT VALIDATION ========================================
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"cell_type_col '{cell_type_col}' not found in adata.obs")
    if isr_matrix_key not in adata.obsm:
        raise ValueError(f"ISR matrix '{isr_matrix_key}' not found in adata.obsm")
    if plot_style not in ['violin', 'box', 'bar', 'strip', 'swarm']:
        raise ValueError("plot_style must be 'violin', 'box', 'bar', 'strip', or 'swarm'")

    # ======================================== GET ISR MATRIX ========================================
    isr_mtx = adata.obsm[isr_matrix_key]
    if not isinstance(isr_mtx, pd.DataFrame):
        regulon_names = adata.uns.get('regulon_names', [f"Regulon_{i}" for i in range(isr_mtx.shape[1])])
        isr_mtx = pd.DataFrame(isr_mtx, index=adata.obs_names, columns=regulon_names)

    # ======================================== REGULON SELECTION ========================================
    if selected_regulons is None:
        selected_regulons = list(isr_mtx.columns)
    elif isinstance(selected_regulons, str):
        selected_regulons = [selected_regulons]
    available_regulons = set(isr_mtx.columns).intersection(selected_regulons)
    if not available_regulons:
        raise ValueError(f"None of the selected regulons found in ISR matrix")
    isr_mtx = isr_mtx[list(available_regulons)]
    print(f"Testing {len(available_regulons)} regulon(s): {list(available_regulons)}")

    # ======================================== CELL TYPE FILTERING ========================================
    if selected_cell_types is not None and excluded_cell_types is not None:
        raise ValueError("Cannot specify both selected_cell_types and excluded_cell_types.")
    
    cell_type_labels = adata.obs[cell_type_col].copy()
    if selected_cell_types is not None:
        mask = cell_type_labels.isin(selected_cell_types)
        cell_types_filtered = selected_cell_types
        print(f"Selected cell types: {selected_cell_types}")
    elif excluded_cell_types is not None:
        mask = ~cell_type_labels.isin(excluded_cell_types)
        cell_types_filtered = [ct for ct in cell_type_labels.unique() if ct not in excluded_cell_types]
        print(f"Excluded cell types: {excluded_cell_types}")
    else:
        mask = pd.Series(True, index=cell_type_labels.index)
        cell_types_filtered = sorted(cell_type_labels.unique())
    
    cell_type_labels = cell_type_labels[mask]
    isr_mtx = isr_mtx.loc[mask]
    n_groups = len(cell_types_filtered)
    if n_groups < 2:
        raise ValueError(f"Need at least 2 cell types for comparison, found {n_groups}")
    print(f"Comparing {n_groups} cell types: {cell_types_filtered}")

    # ======================================== TEST SELECTION ========================================
    if test.lower() == 'auto':
        test = 'kruskal' if n_groups > 2 else 'mannwhitney'
        test_display = 'Kruskal-Wallis' if n_groups > 2 else 'Mann-Whitney U'
    else:
        test_display = test.capitalize()

    # ======================================== LONG FORMAT + GROUP MEANS ========================================
    long_df = pd.melt(isr_mtx.reset_index(), id_vars='index', value_vars=isr_mtx.columns,
                      var_name='Regulon', value_name='ISR Score')
    long_df = long_df.set_index('index').join(cell_type_labels).reset_index(drop=True)
    
    # Pre-compute mean per group (used for fold-change)
    group_means = long_df.groupby(['Regulon', cell_type_col])['ISR Score'].mean().unstack(fill_value=0)

    # ======================================== STATISTICAL TESTING ========================================
    overall_results = []
    pairwise_results = []

    for regulon in available_regulons:
        data = long_df[long_df['Regulon'] == regulon]
        groups = [data[data[cell_type_col] == ct]['ISR Score'].values for ct in cell_types_filtered]
        means = group_means.loc[regulon]

        # Overall test
        try:
            if test == 'mannwhitney':
                stat, p = mannwhitneyu(groups[0], groups[1])
            elif test == 'kruskal':
                stat, p = kruskal(*groups)
            elif test == 'anova':
                stat, p = f_oneway(*groups)
            elif test == 'ttest':
                stat, p = ttest_ind(groups[0], groups[1])
            overall_results.append({'regulon': regulon, 'test': test_display, 'statistic': stat, 'p_value': p})
        except Exception as e:
            warnings.warn(f"Overall test failed for {regulon}: {e}")

        # Pairwise post-hoc (with fold-change)
        if post_hoc:
            for ct1, ct2 in combinations(cell_types_filtered, 2):
                g1 = data[data[cell_type_col] == ct1]['ISR Score']
                g2 = data[data[cell_type_col] == ct2]['ISR Score']
                if len(g1) == 0 or len(g2) == 0:
                    continue
                m1, m2 = means[ct1], means[ct2]
                if m1 == 0 or m2 == 0:
                    log2fc = np.nan
                    fc = np.nan
                else:
                    log2fc = np.log2(m2 / m1)
                    fc = m2 / m1
                _, p_pair = mannwhitneyu(g1, g2)
                pairwise_results.append({
                    'regulon': regulon,
                    'group1': ct1,
                    'group2': ct2,
                    'mean_group1': round(m1, 5),
                    'mean_group2': round(m2, 5),
                    'log2FC': round(log2fc, 3) if not np.isnan(log2fc) else np.nan,
                    'fold_change': round(fc, 3) if not np.isnan(fc) else np.nan,
                    'direction': 'up' if m2 > m1 else 'down',
                    'p_value': p_pair
                })

    # ======================================== CORRECT P-VALUES ========================================
    overall_df = pd.DataFrame(overall_results)
    if not overall_df.empty:
        overall_df['p_adjusted'] = smm.multipletests(overall_df['p_value'], method='bonferroni')[1]
        overall_df['significant'] = overall_df['p_adjusted'] < 0.05
        overall_df['sig_symbol'] = overall_df['p_adjusted'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        )

    pairwise_df = pd.DataFrame(pairwise_results)
    if not pairwise_df.empty:
        pairwise_df['p_adjusted'] = smm.multipletests(pairwise_df['p_value'], method='bonferroni')[1]
        pairwise_df['significant'] = pairwise_df['p_adjusted'] < 0.05
        pairwise_df['sig_symbol'] = pairwise_df['p_adjusted'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        )
        pairwise_df = pairwise_df.sort_values('p_adjusted').reset_index(drop=True)

    # ======================================== PRINT ENHANCED TABLE ========================================
    if show_stats_table:
        print("\n" + "="*110)
        print("ISR COMPARISON WITH FOLD-CHANGE")
        print("="*110)
        if not overall_df.empty:
            print(overall_df[['regulon','test','statistic','p_value','p_adjusted','sig_symbol']].to_string(index=False))
        if not pairwise_df.empty:
            print("\nPAIRWISE COMPARISONS (sorted by significance):")
            print(pairwise_df[['regulon','group1','group2','mean_group1','mean_group2',
                              'log2FC','fold_change','p_adjusted','sig_symbol']].to_string(index=False))

    # ======================================== PLOTTING ========================================
    n_reg = len(available_regulons)
    fig, axes = plt.subplots(1, n_reg, figsize=(figsize[0]//2*n_reg if n_reg>1 else figsize[0], figsize[1]), sharey=False)
    if n_reg == 1: axes = [axes]
    axes = axes.flatten() if n_reg > 1 else axes

    for idx, regulon in enumerate(available_regulons):
        ax = axes[idx]
        df_plot = long_df[long_df['Regulon'] == regulon]

        if plot_style == 'violin':
            sns.violinplot(data=df_plot, x=cell_type_col, y='ISR Score', order=cell_types_filtered,
                           palette=palette, inner=inner, split=split and n_groups==2, ax=ax, alpha=alpha)
            if jitter: sns.stripplot(data=df_plot, x=cell_type_col, y='ISR Score', order=cell_types_filtered,
                                     color='black', alpha=0.3, size=2, ax=ax)
            ax.set_xlim(-0.5, n_groups - 0.5)
            
        elif plot_style == 'box':
            sns.boxplot(data=df_plot, x=cell_type_col, y='ISR Score', order=cell_types_filtered,
                        palette=palette, ax=ax, boxprops=dict(alpha=alpha))
            if jitter: sns.stripplot(data=df_plot, x=cell_type_col, y='ISR Score', order=cell_types_filtered,
                                     color='black', alpha=0.3, size=2, ax=ax)
        elif plot_style == 'bar':
            means = df_plot.groupby(cell_type_col)['ISR Score'].mean().reindex(cell_types_filtered)
            sems = df_plot.groupby(cell_type_col)['ISR Score'].sem().reindex(cell_types_filtered)
            ax.bar(range(len(means)), means, yerr=sems, capsize=5, color=[palette.get(c,'gray') for c in means.index], alpha=alpha)
            if jitter:
                for i, ct in enumerate(cell_types_filtered):
                    subset = df_plot[df_plot[cell_type_col]==ct]['ISR Score']
                    x_j = np.random.normal(i, 0.04, len(subset))
                    ax.scatter(x_j, subset, color='black', alpha=0.3, s=20)

        # Significance lines (unchanged logic, just uses new pairwise_df)
        if show_asterisk and line_asterisk and not pairwise_df.empty:
            sig_pairs = pairwise_df[(pairwise_df['regulon']==regulon) & (pairwise_df['significant'])]
            if not sig_pairs.empty and len(sig_pairs) <= max_comparisons_to_show:
                y_max = df_plot['ISR Score'].max()
                y_range = y_max - df_plot['ISR Score'].min()
                pos_map = {ct:i for i,ct in enumerate(cell_types_filtered)}
                for offset, row in enumerate(sig_pairs.itertuples()):
                    x1, x2 = pos_map[row.group1], pos_map[row.group2]
                    y_pos = y_max + y_range * (0.08 + offset*0.12)
                    ax.plot([x1, x2], [y_pos, y_pos], 'k-', lw=1.5)
                    ax.text((x1+x2)/2, y_pos-0.04, row.sig_symbol, ha='center', va='bottom',
                            fontsize=asterisk_size, fontweight='bold')
                ax.set_ylim(ax.get_ylim()[0], y_max + y_range * (0.15 + offset*0.12))

        ax.set_xlabel(cell_type_col, fontsize=fontsize_xlabel)
        ax.set_ylabel(f'{regulon} ISR Score', fontsize=fontsize_ylabel)
        ax.tick_params(axis='x', rotation=rotation_xtick, labelsize=fontsize_xtick)
        plt.setp(ax.get_xticklabels(), ha='right')
        ax.tick_params(axis='y', labelsize=fontsize_ytick)
        ax.set_title(regulon, fontsize=fontsize_title)

    plt.tight_layout()
    if plot_dir:
        import os
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/isr_test_{plot_style}_{'_'.join(available_regulons)}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    # ======================================== RETURN ========================================
    return {
        'overall_stats': overall_df,
        'pairwise_stats': pairwise_df,   # Now contains log2FC, fold_change, means!
        'group_means': group_means,
        'plot_data': long_df
    }



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
    cbar.set_label(f'{transcription_factor} AUC', fontsize=18)  # Colorbar label font size
    
    # Set title and labels with larger title font size
    plt.title(title_prefix + (f' ({subset_column}: {sample})' if subset else ''), fontsize=20)
    
    # Save plot
    plt.show()

# Example usage:
"""
# For averaged cell type heatmap with selected regulons
heatmap_data = isr_heatmap(
    PDAC,
    cluster_label='niche3',
    isr_mtx=PDAC.obsm['isr'],
    rss_df=PDAC.uns['rss'],
    selected_regulons=['ETS1(+)', 'FOS(+)', 'FOXP3(+)', 'GATA3(+)', 'IRF3(+)', 'IRF4(+)', 
                       'JUN(+)', 'JUNB(+)', 'MAF(+)', 'NFKB1(+)', 'NR3C1(+)', 'PPARA(+)', 
                       'PPARG(+)', 'RELA(+)', 'RUNX3(+)', 'SMAD2(+)', 'SMAD3(+)', 
                       'STAT1(+)', 'STAT4(+)', 'STAT6(+)', 'TFEB(+)', 'TP53(+)'],  
    excluded_cell_types=['Plasma cell'],
    figsize=(20, 7),
    cmap="YlGnBu",
    xticklabels=True,
    yticklabels=True,
    row_cluster=True,
    col_cluster=True
)
"""


# Named tuple to store marker regulon results
# Using namedtuple to avoid autoreload issues with custom __repr__
from collections import namedtuple
MarkerRegulonResult = namedtuple('MarkerRegulonResult', ['dataframe', 'marker_dict'])

def _plot_ranked_marker_regulons(adata_auc, markers_df, groupby, n_genes=10, method='wilcoxon'):
    """
    Create a ranked scatter plot showing top marker regulons for each group.
    Similar to scanpy's sc.pl.rank_genes_groups() visualization but with enhanced styling.
    
    THIS FUNCTION INCLUDES TEXT REPELLING - Install adjustText for best results:
        pip install adjustText
    
    Parameters
    ----------
    adata_auc : AnnData
        AnnData object with AUCell scores
    markers_df : pd.DataFrame
        DataFrame from rank_genes_groups with columns: ['group', 'regulon', 'scores', etc.]
    groupby : str
        Column name used for grouping
    n_genes : int
        Number of top marker regulons to show per group
    method : str
        Statistical test method used
    """
    import matplotlib.pyplot as plt
    
    # ============================================================
    # REPEL FUNCTIONALITY: Try to import adjustText
    # This library automatically adjusts text positions to avoid overlap
    # Install with: pip install adjustText
    # ============================================================
    try:
        from adjustText import adjust_text
        use_adjust_text = True
        print("  ✓ Using adjustText for automatic label repelling")
    except ImportError:
        use_adjust_text = False
        print("  ⚠ adjustText not found. Labels may overlap.")
        print("  Install for better label positioning: pip install adjustText")
    
    # Get unique groups
    groups = sorted(markers_df['group'].unique())
    n_groups = len(groups)
    
    # Determine score column to use
    score_col = 'scores' if 'scores' in markers_df.columns else 'logfoldchanges'
    
    # Create subplots - one per group
    n_cols = min(3, n_groups)
    n_rows = int(np.ceil(n_groups / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
    
    # Flatten axes array for easier iteration
    if n_groups == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Plot each group
    for idx, group in enumerate(groups):
        ax = axes[idx]
        
        # Get top N marker regulons for this group
        group_data = markers_df[markers_df['group'] == group].head(n_genes)
        
        # Prepare data for plotting
        regulons = group_data['regulon'].tolist()
        scores = group_data[score_col].tolist()
        rankings = list(range(len(regulons)))
        
        # Create scatter plot
        ax.scatter(rankings, scores, s=150, alpha=0.6, edgecolors='black', linewidth=0.8, zorder=3)
        
        # ============================================================
        # ADD TEXT LABELS - These will be adjusted to avoid overlap
        # ============================================================
        texts = []
        for rank, regulon, score in zip(rankings, regulons, scores):
            txt = ax.text(rank, score, regulon, fontsize=16, ha='center', va='bottom', 
                         fontweight='bold', zorder=4)
            texts.append(txt)
        
        # ============================================================
        # APPLY TEXT REPELLING 
        # The adjust_text function moves labels to avoid overlaps
        # ============================================================
        if use_adjust_text and len(texts) > 0:
            adjust_text(
                texts,                          # List of text objects to adjust
                ax=ax,                          # The axes to work on
                arrowprops=dict(                # Draw arrows from label to point
                    arrowstyle='->',
                    color='black',
                    lw=1,
                    alpha=0.9
                ),
                expand_points=(1.5, 1.5),       # Space around points
                expand_text=(1.2, 1.2),         # Space around text
                force_points=(0.5, 0.5),        # Repulsion force from points
                force_text=(0.5, 0.5),          # Repulsion force between texts
                only_move={'points': 'y', 'texts': 'xy'}  # Allow text to move in x and y
            )
            
        elif not use_adjust_text and len(texts) > 0:
            print(f"    ⚠ No repelling applied for group (adjustText not installed, pip install adjustText)")
        
        # Customize axes
        ax.set_xlabel('rank', fontsize=20, fontweight='bold')
        ax.set_ylabel('AUCell ranked score', fontsize=20, fontweight='bold')
        ax.set_title(f'{group} vs. rest', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.tick_params(axis='both', labelsize=20) 
        
        # Set x-axis limits with padding
        if len(rankings) > 0:
            ax.set_xlim(-0.5, max(rankings) + 0.5)
        
        # Add extra padding to y-axis for labels
        if len(scores) > 0:
            y_range = max(scores) - min(scores)
            y_padding = y_range * 0.15
            ax.set_ylim(min(scores) - y_padding, max(scores) + y_padding)
    
    # Hide extra subplots if any
    for idx in range(n_groups, len(axes)):
        axes[idx].axis('off')
    
    # Add overall title
    fig.suptitle(f'Top {n_genes} Marker Regulons per Group ({method})', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.show()
    
    print(f"  ✓ Generated ranked scatter plot with top {n_genes} markers per group")



def find_marker_regulon(
    adata,
    groupby=None,
    auc_matrix_key='auc_mtx',  # NEW PARAMETER
    method='wilcoxon',
    top_n=10,
    output_dir=None,
    plot='dotplot',
    resolutions=None,
    ranked_n_genes=10
):
    """
    Find marker regulons for clusters based on AUCell scores.
    Can auto-cluster using Leiden on the specified AUCell matrix if groupby is None.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with AUCell results.
    groupby : str or None, optional (default: None)
        Column name in adata.obs to group by (e.g., 'leiden', 'cell_type', 'grn_res_0.5')
        If None, will compute Leiden clustering on AUCell matrix first
    auc_matrix_key : str, optional (default: 'auc_mtx')
        Key in adata.obsm for the AUCell matrix to use (e.g., 'auc_mtx', 'isr')
        This allows using different AUCell matrices for different analyses
    method : str, optional (default: 'wilcoxon')
        Statistical test to use. Options: 'wilcoxon', 't-test', 't-test_overestim_var', 'logreg'
    top_n : int, optional (default: 10)
        Number of top marker regulons to show per cluster
    output_dir : str or None, optional (default: None)
        Directory path to save the marker regulons CSV file. If None, file is not saved
    plot : str, list of str, or None, optional (default: 'dotplot')
        Type of plot(s) to generate. Options: 'dotplot', 'heatmap', 'matrixplot', 'stacked_violin', 'ranked', None
        Can be a single string or list of strings to generate multiple plots
    resolutions : list, float, or None, optional (default: None)
        Resolution(s) for Leiden clustering. Only used if groupby=None.
        If None, defaults to [0.5, 1.0, 1.5]
        Can be a single float or list of floats
    ranked_n_genes : int, optional (default: 10)
        Number of top marker regulons to show per group in ranked plot.
        Only used when plot includes 'ranked'.
        
    Returns
    -------
    MarkerRegulonResult or dict
        Single result if groupby provided, dict of results if auto-clustering
                           
    Examples
    --------
    # Use default AUCell matrix ('auc_mtx')
    result = find_marker_regulon(adata, groupby='cell_type', plot='ranked')
    
    # Use ISR matrix instead
    result = find_marker_regulon(adata, groupby='cell_type', auc_matrix_key='isr', plot='dotplot')
    
    # Auto-clustering with custom matrix
    results = find_marker_regulon(adata, auc_matrix_key='isr', resolutions=[0.5, 1.0])
    
    # Multiple plot types
    result = find_marker_regulon(
        adata, 
        groupby='leiden', 
        auc_matrix_key='auc_mtx',
        plot=['dotplot', 'ranked', 'heatmap']
    )
    """
    
    print("="*60)
    print("Finding Marker Regulons")
    print("="*60)
    
    # Check if specified AUCell matrix exists
    if auc_matrix_key not in adata.obsm:
        raise ValueError(
            f"AUCell matrix not found in adata.obsm['{auc_matrix_key}']. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    
    print(f"\nUsing AUCell matrix: '{auc_matrix_key}'")
    
    # Get the AUCell matrix
    auc_mtx = adata.obsm[auc_matrix_key]
    
    # Convert to DataFrame if needed
    if not isinstance(auc_mtx, pd.DataFrame):
        # Try to get regulon names
        if 'regulon_names' in adata.uns:
            regulon_names = adata.uns['regulon_names']
        else:
            # Generate default names
            regulon_names = [f"Regulon_{i}" for i in range(auc_mtx.shape[1])]
        
        auc_mtx = pd.DataFrame(
            auc_mtx,
            index=adata.obs_names,
            columns=regulon_names
        )
    
    print(f"  - Matrix shape: {auc_mtx.shape[0]} cells x {auc_mtx.shape[1]} regulons")
    
    # Handle case when groupby is None - compute clustering on AUCell matrix
    if groupby is None:
        print("\nNo groupby provided. Computing Leiden clustering on AUCell matrix...")
        
        # Check if UMAP for this matrix exists
        umap_key = f"X_umap_{auc_matrix_key}"
        if umap_key not in adata.obsm:
            print(f"Computing neighbors and UMAP on '{auc_matrix_key}' matrix...")
            
            # Create temporary AnnData for clustering
            ad_auc = sc.AnnData(auc_mtx)
            
            # Compute neighbors and UMAP
            sc.pp.neighbors(ad_auc, n_neighbors=10, metric="correlation")
            sc.tl.umap(ad_auc)
            
            # Store in original AnnData
            adata.obsm[umap_key] = ad_auc.obsm["X_umap"]
            
            print(f"✓ Added UMAP based on '{auc_matrix_key}' scores")
        else:
            print(f"Using existing UMAP coordinates from '{umap_key}'")
            
            # Create temporary AnnData for clustering
            ad_auc = sc.AnnData(auc_mtx)
            
            # Need to recompute neighbors for clustering
            print(f"Computing neighbors on '{auc_matrix_key}' matrix for clustering...")
            sc.pp.neighbors(ad_auc, n_neighbors=10, metric="correlation")
        
        # Handle resolutions
        if resolutions is None:
            resolutions = [0.5, 1.0, 1.5]
        elif isinstance(resolutions, (int, float)):
            resolutions = [float(resolutions)]
        else:
            resolutions = [float(r) for r in resolutions]
        
        print(f"\nComputing Leiden clustering on '{auc_matrix_key}' matrix for resolutions: {resolutions}")
        
        # Compute Leiden clustering for all resolutions
        clustering_keys = []
        for res in resolutions:
            key = f'{auc_matrix_key}_res_{res}'
            print(f"\nComputing Leiden clustering with resolution {res}...")
            sc.tl.leiden(ad_auc, resolution=res, key_added=key)
            n_clusters = ad_auc.obs[key].nunique()
            print(f"  Found {n_clusters} clusters")
            
            # Transfer clustering to original adata
            adata.obs[key] = ad_auc.obs[key].values
            
            # Skip if only 1 cluster (no differential analysis possible)
            if n_clusters <= 1:
                print(f"  WARNING: Only {n_clusters} cluster(s) found. Skipping marker regulon analysis for this resolution.")
                continue
            
            clustering_keys.append(key)
        
        if not clustering_keys:
            raise ValueError("No valid clusterings with >1 cluster found. Try different resolutions.")
        
        # Run marker regulon analysis for each clustering
        results = {}
        for key in clustering_keys:
            print(f"\n{'='*60}")
            print(f"Finding marker regulons for '{key}'")
            print(f"{'='*60}\n")
            
            # Adjust output directory for multiple clusterings
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                current_output = output_dir
            else:
                current_output = None
            
            markers_df, marker_dict = _find_markers_for_groupby(
                adata=adata,
                groupby=key,
                auc_mtx=auc_mtx,
                method=method,
                top_n=top_n,
                output_dir=current_output,
                plot=plot,
                ranked_n_genes=ranked_n_genes
            )
            
            results[key] = MarkerRegulonResult(markers_df, marker_dict)
        
        print("\n" + "="*80)
        print("✓ Marker Regulon Analysis Completed!")
        print("="*80)
        print(f"\nClustering results stored in adata.obs:")
        for key in clustering_keys:
            print(f"  - adata.obs['{key}']")
        
        return results
    
    else:
        # Standard case - groupby is provided
        if groupby not in adata.obs.columns:
            raise ValueError(
                f"Provided groupby '{groupby}' not found in adata.obs. "
                f"Available keys: {adata.obs.columns.tolist()}"
            )
        
        # Check if groupby has more than 1 unique value
        n_groups = adata.obs[groupby].nunique()
        if n_groups <= 1:
            raise ValueError(
                f"'{groupby}' has only {n_groups} unique value(s). "
                "Need at least 2 groups for marker regulon analysis."
            )
        
        print(f"\nUsing existing grouping: '{groupby}'")
        print(f"  - Number of groups: {n_groups}")
        
        markers_df, marker_dict = _find_markers_for_groupby(
            adata=adata,
            groupby=groupby,
            auc_mtx=auc_mtx,
            method=method,
            top_n=top_n,
            output_dir=output_dir,
            plot=plot,
            ranked_n_genes=ranked_n_genes
        )
        print("")  # Add extra line for cleaner output
        
        return MarkerRegulonResult(markers_df, marker_dict)


def _find_markers_for_groupby(
    adata, groupby, auc_mtx, method, top_n, output_dir, plot, ranked_n_genes
):
    """
    Internal function to find marker regulons for a specific grouping.
    This is the same as in the existing plot_Regulon.py code.
    """
    # Create temporary AnnData
    ad_plot = sc.AnnData(auc_mtx)
    ad_plot.obs[groupby] = adata.obs[groupby].values
    
    # Run differential analysis
    print(f"Running sc.tl.rank_genes_groups with method='{method}'...")
    sc.tl.rank_genes_groups(
        ad_plot,
        groupby=groupby,
        method=method,
        key_added='rank_genes_groups'
    )
    
    # Extract results
    result = ad_plot.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    marker_df = pd.DataFrame()
    marker_dict = {}
    
    for group in groups:
        names = result['names'][group][:top_n]
        scores = result['scores'][group][:top_n] if 'scores' in result else None
        pvals = result['pvals'][group][:top_n] if 'pvals' in result else None
        pvals_adj = result['pvals_adj'][group][:top_n] if 'pvals_adj' in result else None
        logfoldchanges = result['logfoldchanges'][group][:top_n] if 'logfoldchanges' in result else None
        
        marker_dict[group] = list(names)
        
        temp_df = pd.DataFrame({
            'group': [group] * len(names),
            'regulon': names,
            'scores': scores if scores is not None else np.nan,
            'pvals': pvals if pvals is not None else np.nan,
            'pvals_adj': pvals_adj if pvals_adj is not None else np.nan,
            'logfoldchanges': logfoldchanges if logfoldchanges is not None else np.nan
        })
        marker_df = pd.concat([marker_df, temp_df], ignore_index=True)
    
    print(f"Found top {top_n} markers for {len(groups)} groups")
    
    # Save CSV
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"marker_regulons_{groupby}_{method}.csv")
        marker_df.to_csv(csv_path, index=False)
        print(f"Saved marker regulons to {csv_path}")
    
    # PLOTTING - INLINE
    if plot:
        if not isinstance(plot, list):
            plot = [plot]
        
        all_top_regulons = list(set([r for l in marker_dict.values() for r in l]))
        
        for p in plot:
            print(f"\nGenerating {p} plot...")
            
            # HEATMAP - individual cells
            if p == 'heatmap':
                sampled_cells = []
                for ct in sorted(adata.obs[groupby].unique()):
                    ct_cells = adata.obs_names[adata.obs[groupby] == ct]
                    if len(ct_cells) > 50:
                        ct_cells = np.random.choice(ct_cells, 50, replace=False)
                    sampled_cells.extend(ct_cells)
                
                plot_data = auc_mtx.loc[sampled_cells, all_top_regulons]
                
                sampled_cell_types = adata.obs[groupby].loc[sampled_cells]
                unique_types = sorted(sampled_cell_types.unique())
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
                type_colors = dict(zip(unique_types, colors))
                row_colors = pd.Series([type_colors[ct] for ct in sampled_cell_types],
                                      index=sampled_cells, name='Cell Type')
                
                g = sns.clustermap(plot_data,
                                  row_colors=row_colors,
                                  figsize=(15, 10),
                                  yticklabels=False,
                                  row_cluster=True,
                                  col_cluster=True
                )
                
                g.ax_heatmap.set_ylabel('Individual Cells', fontsize=18)
                g.ax_heatmap.set_xlabel('Regulons', fontsize=18)
                g.ax_heatmap.yaxis.tick_left()
                g.ax_heatmap.yaxis.set_label_position('left')
                
                plt.title(f'Individual Cell ISR Heatmap ({len(sampled_cells)} cells, {len(plot_data.columns)} regulons)', 
                          fontsize=18, pad=20)
                
                plt.tight_layout()
                plt.show()
            
            # VIOLIN
            elif p in ['violin', 'stacked_violin']:
                long_df = pd.melt(
                    auc_mtx[all_top_regulons].reset_index(),
                    id_vars=['index'],
                    value_vars=all_top_regulons,
                    var_name='Regulon',
                    value_name='ISR Score'
                )
                long_df = long_df.set_index('index')
                long_df = long_df.join(adata.obs[[groupby]])
                
                g = sns.catplot(
                    data=long_df,
                    x=groupby,
                    y='ISR Score',
                    col='Regulon' if len(all_top_regulons) > 1 else None,
                    col_wrap=4 if len(all_top_regulons) > 1 else None,
                    kind='violin',
                    height=6,
                    aspect=1
                )
                
                for ax in g.axes.flat:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
                    ax.yaxis.tick_left()
                    ax.yaxis.set_label_position('left')
                
                g.fig.suptitle(f'ISR Violin Plots ({len(all_top_regulons)} regulons)', y=1.02, fontsize=16)
                g.tight_layout()
                plt.show()
            
            # DOTPLOT
            elif p == 'dotplot':
                sc.pl.dotplot(ad_plot, var_names=all_top_regulons, groupby=groupby, 
                              use_raw=False, show=False)
                ax = plt.gca()
                ax.yaxis.tick_left()
                ax.yaxis.set_label_position('left')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', va='center')
                plt.show()
            
            # MATRIXPLOT
            elif p == 'matrixplot':
                sc.pl.matrixplot(ad_plot, var_names=all_top_regulons, groupby=groupby, 
                                 use_raw=False, show=False)
                ax = plt.gca()
                ax.yaxis.tick_left()
                ax.yaxis.set_label_position('left')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', va='center')
                plt.show()
            
            # RANKED
            elif p == 'ranked':
                _plot_ranked_marker_regulons(
                    ad_plot,
                    marker_df,
                    groupby=groupby,
                    n_genes=ranked_n_genes,
                    method=method
                )
    
    return marker_df, marker_dict




