import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import seaborn as sns
from matplotlib.patches import Rectangle
import scanpy as sc
from scipy.sparse import issparse
from .hotspot import Hotspot
from .local_stats_pairs import compute_hs_pairs_centered_cond
import pickle
import os
import warnings



def plot_grn_network(
    adata,
    receptor_source: str = 'receptor_dict',
    min_importance: float = 0.1,
    max_nodes: Optional[int] = None,
    selected_tfs: Optional[List[str]] = None,
    selected_receptors: Optional[List[str]] = None,
    show_receptor: bool = True,
    layout_type: str = 'layered',
    node_size_scale: float = 1.0,
    edge_width_scale: float = 1.0,
    edge_length_scale: float = 1.0,
    show_edge_labels: bool = False,
    color_scheme: str = 'default',
    figsize: Tuple[int, int] = (8, 7),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
    font_size: int = 12,
    dpi: int = 300
) -> Tuple[plt.Figure, nx.DiGraph]:
    """
    Create a Gene Regulatory Network plot showing receptors -> TFs -> target genes.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing GRN analysis results
    receptor_source : str, default 'receptor_dict'
        Key in adata.uns for receptor data. Only 'receptor_dict' is supported.
    min_importance : float, default 0.1
        Minimum edge importance threshold for TF-target connections
    max_nodes : int, optional
        Maximum number of nodes to display (prioritizes by importance)
    selected_tfs : list of str, optional
        Specific TFs to include. If None, includes all TFs
    selected_receptors : list of str, optional
        Specific receptors to include. If None, includes all receptors
    show_receptor : bool, default True
        Whether to include receptor nodes and receptor-to-TF edges
    layout_type : str, default 'layered'
        Network layout type: 'layered', 'spring', 'circular', 'hierarchical'
    node_size_scale : float, default 1.0
        Scaling factor for node sizes (not adjusted per user request)
    edge_width_scale : float, default 1.0
        Scaling factor for edge widths
    edge_length_scale : float, default 1.0
        Scaling factor for edge lengths/node spacing. Higher values increase spacing.
    show_edge_labels : bool, default False
        Whether to show importance values on edges
    color_scheme : str, default 'default'
        Color scheme: 'default', 'pastel', 'bright', 'earth'
    figsize : tuple, default (8, 7)
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot
    title : str, optional
        Plot title. If None, defaults to 'Gene Regulatory Network (TFs to Targets)' if show_receptor=False,
        else 'Gene Regulatory Network ({receptor_source})'
    show_legend : bool, default True
        Whether to show the legend
    font_size : int, default 12
        Font size for labels (not adjusted per user request)
    dpi : int, default 300
        DPI for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    G : networkx.DiGraph
        The network graph object
    """

    # Validate inputs
    if receptor_source != 'receptor_dict':
        raise ValueError("receptor_source must be 'receptor_dict'. receptor_dict_all is not supported.")

    # Extract data from adata
    try:
        receptor_data = adata.uns.get(receptor_source, {}) if show_receptor else {}
        regulon_dict = adata.uns.get('regulon_dict', {})
        adj_data = adata.uns.get('adj', [])

        if isinstance(adj_data, pd.DataFrame):
            adj_data = adj_data.to_dict('records')

    except KeyError as e:
        raise KeyError(f"Required data not found in adata.uns: {e}")

    # Validate data availability
    if show_receptor and not receptor_data:
        warnings.warn(f"No receptor data found in adata.uns['{receptor_source}']")
        return None, None

    if not regulon_dict:
        warnings.warn("No regulon data found in adata.uns['regulon_dict']")
        return None, None

    if not adj_data:
        warnings.warn("No adjacency data found in adata.uns['adj']")
        return None, None

    # Filter data based on selections
    if selected_tfs:
        if show_receptor:
            receptor_data = {tf: receptors for tf, receptors in receptor_data.items() if tf in selected_tfs}
        regulon_dict = {tf: targets for tf, targets in regulon_dict.items()
                       if tf.replace('(+)', '') in selected_tfs}

    # Filter adjacency data by importance and selected TFs
    filtered_adj = []
    for edge in adj_data:
        if edge.get('importance', 0) >= min_importance:
            tf_name = edge.get('TF', '')
            if not selected_tfs or tf_name in selected_tfs:
                filtered_adj.append(edge)

    # Build network graph
    G = nx.DiGraph()

    # Add nodes and edges
    receptor_nodes = set()
    tf_nodes = set()
    target_nodes = set()

    # Add receptor-TF connections only if show_receptor is True
    if show_receptor:
        for tf, receptors in receptor_data.items():
            if selected_receptors:
                receptors = [r for r in receptors if r in selected_receptors]
            tf_nodes.add(tf)
            for receptor in receptors:
                receptor_nodes.add(receptor)
                G.add_edge(receptor, tf, edge_type='receptor_tf', weight=1.0)

    # Add TF-target connections from adjacency data
    tf_target_edges = {}
    for edge in filtered_adj:
        tf = edge.get('TF', '')
        target = edge.get('target', '')
        importance = edge.get('importance', 0)

        # Check if this TF-target pair exists in regulon_dict
        tf_key = f"{tf}(+)"
        if tf_key in regulon_dict and target in regulon_dict[tf_key]:
            tf_nodes.add(tf)
            target_nodes.add(target)
            G.add_edge(tf, target, edge_type='tf_target', weight=importance)
            tf_target_edges[(tf, target)] = importance

    # Limit nodes if specified
    if max_nodes and len(G.nodes()) > max_nodes:
        node_scores = {}
        for node in G.nodes():
            if node in tf_nodes:
                score = G.degree(node) * 10
            elif node in receptor_nodes:
                score = G.degree(node) * 5
            else:
                score = sum(G[pred][node]['weight'] for pred in G.predecessors(node))
            node_scores[node] = score

        top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        keep_nodes = [node for node, _ in top_nodes]
        G = G.subgraph(keep_nodes).copy()

        receptor_nodes = receptor_nodes.intersection(keep_nodes)
        tf_nodes = tf_nodes.intersection(keep_nodes)
        target_nodes = target_nodes.intersection(keep_nodes)

    # Define color schemes
    color_schemes = {
        'default': {
            'receptor': '#e74c3c',
            'tf': '#f39c12',
            'target': '#3498db',
            'receptor_tf_edge': '#e74c3c',
            'tf_target_edge': '#3498db'
        },
        'pastel': {
            'receptor': '#ffb3ba',
            'tf': '#ffffba',
            'target': '#bae1ff',
            'receptor_tf_edge': '#ffb3ba',
            'tf_target_edge': '#bae1ff'
        },
        'bright': {
            'receptor': '#ff0000',
            'tf': '#ffff00',
            'target': '#0000ff',
            'receptor_tf_edge': '#ff0000',
            'tf_target_edge': '#0000ff'
        },
        'earth': {
            'receptor': '#2E4057',
            'tf': '#048A81',
            'target': '#A73E5C',
            'receptor_tf_edge': '#2E4057',
            'tf_target_edge': '#A73E5C'
        }
    }

    colors = color_schemes[color_scheme]

    # Calculate layout with edge length scaling
    if layout_type == 'layered':
        pos = _calculate_layered_layout(G, receptor_nodes, tf_nodes, target_nodes, edge_length_scale, show_receptor)
        if not pos or len(pos) == 0:
            print("Warning: Layered layout failed, falling back to spring layout")
            pos = nx.spring_layout(G, k=1/(max(len(G.nodes())**0.5, 1)) * edge_length_scale, iterations=50)
    elif layout_type == 'spring':
        pos = nx.spring_layout(G, k=1/(max(len(G.nodes())**0.5, 1)) * edge_length_scale, iterations=50)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
        pos = {node: (x * edge_length_scale, y * edge_length_scale) for node, (x, y) in pos.items()}
    elif layout_type == 'hierarchical':
        receptor_list = list(receptor_nodes) if receptor_nodes and show_receptor else []
        tf_list = list(tf_nodes) if tf_nodes else []
        target_list = list(target_nodes) if target_nodes else []
        node_lists = [lst for lst in [receptor_list, tf_list, target_list] if lst]
        if node_lists:
            pos = nx.shell_layout(G, nlist=node_lists)
            pos = {node: (x * edge_length_scale, y * edge_length_scale) for node, (x, y) in pos.items()}
        else:
            pos = nx.spring_layout(G, k=1/(max(len(G.nodes())**0.5, 1)) * edge_length_scale, iterations=50)
    else:
        pos = nx.spring_layout(G, k=1/(max(len(G.nodes())**0.5, 1)) * edge_length_scale, iterations=50)

    # Normalize positions to fit within figure boundaries
    if pos:
        x_coords = [x for x, y in pos.values()]
        y_coords = [y for x, y in pos.values()]
        if x_coords and y_coords:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_range = x_max - x_min if x_max != x_min else 1
            y_range = y_max - y_min if y_max != y_min else 1

            # Scale positions to [0, 1] range and adjust for figure aspect ratio
            fig_width, fig_height = figsize
            aspect_ratio = fig_width / fig_height
            pos = {
                node: (
                    (x - x_min) / x_range * 0.8 + 0.1,  # Scale to 80% of width, centered
                    (y - y_min) / y_range * 0.8 * aspect_ratio + 0.1  # Adjust for aspect ratio
                ) for node, (x, y) in pos.items()
            }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Draw edges
    receptor_tf_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'receptor_tf']
    tf_target_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'tf_target']

    # Draw receptor-TF edges
    if show_receptor and receptor_tf_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=receptor_tf_edges,
            edge_color=colors['receptor_tf_edge'],
            width=2.0 * edge_width_scale,
            alpha=0.7, ax=ax, arrows=True, arrowsize=20
        )

    # Draw TF-target edges with varying widths based on importance
    if tf_target_edges:
        edge_weights = [G[u][v]['weight'] for u, v in tf_target_edges]
        edge_widths = [1 + w * 3 * edge_width_scale for w in edge_weights]
        edge_alphas = [0.5 + w * 0.5 for w in edge_weights]

        for (u, v), width, alpha in zip(tf_target_edges, edge_widths, edge_alphas):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                edge_color=colors['tf_target_edge'],
                width=width, alpha=alpha, ax=ax, arrows=True, arrowsize=15
            )

    # Draw nodes (node sizes unchanged per user request)
    node_sizes = {
        'receptor': 300 * node_size_scale,
        'tf': 500 * node_size_scale,
        'target': 200 * node_size_scale
    }

    for node_type, nodes, color in [
        ('receptor', receptor_nodes, colors['receptor']) if show_receptor else ('receptor', set(), colors['receptor']),
        ('tf', tf_nodes, colors['tf']),
        ('target', target_nodes, colors['target'])
    ]:
        if nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes,
                node_color=color, node_size=node_sizes[node_type],
                alpha=0.8, ax=ax
            )

    # Draw labels with dynamic offset based on figure size
    label_offset = 0.05 * min(figsize) * edge_length_scale  # Scale offset with figure size
    label_pos = {node: (x, y + label_offset) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=font_size, ax=ax)

    # Draw edge labels if requested
    if show_edge_labels:
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            if d['edge_type'] == 'tf_target':
                edge_labels[(u, v)] = f"{d['weight']:.2f}"
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=font_size-2, ax=ax)

    # Add legend if requested
    if show_legend:
        legend_elements = []
        if show_receptor and receptor_nodes:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['receptor'],
                          markersize=10, label=f'Receptors (n={len(receptor_nodes)})'))
        legend_elements.extend([
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['tf'],
                      markersize=12, label=f'TFs (n={len(tf_nodes)})'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['target'],
                      markersize=8, label=f'Targets (n={len(target_nodes)})'),
        ])
        if show_receptor and receptor_tf_edges:
            legend_elements.append(
                plt.Line2D([0], [0], color=colors['receptor_tf_edge'], linewidth=2,
                          label='Receptor -> TF'))
        if tf_target_edges:
            legend_elements.append(
                plt.Line2D([0], [0], color=colors['tf_target_edge'], linewidth=2,
                          label='TF -> Target'))
        ax.legend(handles=legend_elements, loc='best', fontsize=font_size, bbox_to_anchor=(1.0, 1.0))

    # Set title (if provided)
    if title is None:
        title = f'Gene Regulatory Network ({"TFs to Targets" if not show_receptor else receptor_source.replace("_", " ").title()})'
    if title:
        ax.set_title(title, fontsize=18, fontweight='bold')

    # Adjust plot margins to prevent clipping
    x_coords = [x for x, y in pos.values()] + [x for x, y in label_pos.values()]
    y_coords = [y for x, y in pos.values()] + [y for x, y in label_pos.values()]
    if x_coords and y_coords:
        x_margin = 0.3 * (max(x_coords) - min(x_coords)) + 0.1
        y_margin = 0.3 * (max(y_coords) - min(y_coords)) + 0.1
        ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
        ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)

    # Remove axes
    ax.axis('off')
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")

    # Print categorized nodes
    receptor_list = list(receptor_nodes)
    tf_list = list(tf_nodes)
    target_list = list(target_nodes)
    print(f"Receptor: {receptor_list}. TF: {tf_list}, Target: {target_list}")

    return fig, G



def _calculate_layered_layout(G, receptor_nodes, tf_nodes, target_nodes, edge_length_scale=1.0, show_receptor=True):
    """Calculate layered layout positions for GRN visualization with adjustable edge spacing."""
    pos = {}

    receptor_list = list(receptor_nodes) if receptor_nodes and show_receptor else []
    tf_list = list(tf_nodes) if tf_nodes else []
    target_list = list(target_nodes) if target_nodes else []

    print(f"Layered layout - Receptors: {len(receptor_list)}, TFs: {len(tf_list)}, Targets: {len(target_list)}")

    # Layer positions with scaling
    layer_spacing = 3.0 * edge_length_scale
    layers = {
        'receptor': {'x': -layer_spacing, 'y_center': 0} if show_receptor else {'x': 0, 'y_center': 0},
        'tf': {'x': 0 if show_receptor else -layer_spacing/2, 'y_center': 0},
        'target': {'x': layer_spacing if show_receptor else layer_spacing/2, 'y_center': 0}
    }

    # Position receptors
    if show_receptor and receptor_list:
        base_spacing = 2.0 * edge_length_scale
        if len(receptor_list) == 1:
            pos[receptor_list[0]] = (layers['receptor']['x'], 0)
        else:
            y_range = 4.0 * edge_length_scale
            y_spacing = y_range / (len(receptor_list) - 1)
            start_y = -y_range / 2
            for i, receptor in enumerate(receptor_list):
                y = start_y + i * y_spacing
                pos[receptor] = (layers['receptor']['x'], y)

    # Position TFs
    if tf_list:
        if len(tf_list) == 1:
            pos[tf_list[0]] = (layers['tf']['x'], 0)
        else:
            y_range = 4.0 * edge_length_scale
            y_spacing = y_range / (len(tf_list) - 1)
            start_y = -y_range / 2
            for i, tf in enumerate(tf_list):
                y = start_y + i * y_spacing
                pos[tf] = (layers['tf']['x'], y)

    # Position targets
    if target_list:
        if len(target_list) == 1:
            pos[target_list[0]] = (layers['target']['x'], 0)
        else:
            y_range = 5.0 * edge_length_scale
            y_spacing = y_range / (len(target_list) - 1)
            start_y = -y_range / 2
            for i, target in enumerate(target_list):
                y = start_y + i * y_spacing
                pos[target] = (layers['target']['x'], y)

    print(f"Generated {len(pos)} positions")
    return pos

def get_network_summary(adata, receptor_source: str = 'receptor_dict') -> Dict:
    """
    Get summary statistics of the GRN network.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing GRN analysis results
    receptor_source : str
        Key in adata.uns for receptor data

    Returns
    -------
    dict
        Dictionary containing network statistics
    """
    try:
        receptor_data = adata.uns.get(receptor_source, {})
        regulon_dict = adata.uns.get('regulon_dict', {})
        adj_data = adata.uns.get('adj', [])

        if isinstance(adj_data, pd.DataFrame):
            adj_data = adj_data.to_dict('records')
    except KeyError:
        return {}

    total_receptors = len(set(receptor for receptors in receptor_data.values() for receptor in receptors))
    total_tfs = len(receptor_data)
    total_targets = len(set(target for targets in regulon_dict.values() for target in targets))
    total_edges = len(adj_data)

    importances = [edge.get('importance', 0) for edge in adj_data if 'importance' in edge]

    summary = {
        'total_receptors': total_receptors,
        'total_tfs': total_tfs,
        'total_targets': total_targets,
        'total_tf_target_edges': total_edges,
        'avg_receptors_per_tf': total_receptors / max(total_tfs, 1),
        'avg_targets_per_tf': total_targets / max(total_tfs, 1),
        'receptor_source': receptor_source
    }

    if importances:
        summary.update({
            'mean_importance': np.mean(importances),
            'median_importance': np.median(importances),
            'min_importance': np.min(importances),
            'max_importance': np.max(importances)
        })

    return summary

def test_grn_plot():
    """Test function with sample data."""
    import anndata as ad

    sample_data = {
        'receptor_dict': {
            'TF1': ['EGFR', 'PDGFRA'],
            'TF2': ['VEGFR2', 'NOTCH1'],
            'TF3': ['TGFBR1']
        },
        'regulon_dict': {
            'TF1(+)': ['Gene1', 'Gene2', 'Gene3', 'Gene4'],
            'TF2(+)': ['Gene3', 'Gene5', 'Gene6'],
            'TF3(+)': ['Gene7', 'Gene8', 'Gene9', 'Gene10']
        },
        'adj': [
            {'TF': 'TF1', 'target': 'Gene1', 'importance': 0.8},
            {'TF': 'TF1', 'target': 'Gene2', 'importance': 0.7},
            {'TF': 'TF1', 'target': 'Gene3', 'importance': 0.6},
            {'TF': 'TF1', 'target': 'Gene4', 'importance': 0.5},
            {'TF': 'TF2', 'target': 'Gene3', 'importance': 0.9},
            {'TF': 'TF2', 'target': 'Gene5', 'importance': 0.8},
            {'TF': 'TF2', 'target': 'Gene6', 'importance': 0.7},
            {'TF': 'TF3', 'target': 'Gene7', 'importance': 0.6},
            {'TF': 'TF3', 'target': 'Gene8', 'importance': 0.8},
            {'TF': 'TF3', 'target': 'Gene9', 'importance': 0.7},
            {'TF': 'TF3', 'target': 'Gene10', 'importance': 0.5}
        ]
    }

    adata = ad.AnnData(np.random.rand(100, 20))
    adata.uns.update(sample_data)

    print("Testing basic usage...")
    fig, G = plot_grn_network(adata, title="Basic GRN Network")
    if fig:
        plt.show()

    print("\nTesting with increased edge spacing...")
    fig, G = plot_grn_network(
        adata,
        edge_length_scale=2.0,
        title="GRN Network - Increased Spacing"
    )
    if fig:
        plt.show()

    print("\nTesting without receptors and custom title...")
    fig, G = plot_grn_network(
        adata,
        show_receptor=False,
        selected_tfs=['TF1', 'TF2'],
        title="Custom TF-to-Target Network",
        show_legend=False
    )
    if fig:
        plt.show()

    summary = get_network_summary(adata, 'receptor_dict')
    print("Network Summary:", summary)

if __name__ == "__main__":
    test_grn_plot()

    
    # Example usage:
    # Focus on specific TFs without receptors, custom title, and no legend
    # fig, G = plot_grn_network(
    #     adata,
    #     receptor_source='receptor_dict',
    #     selected_tfs=['IRF3'],
    #     show_receptor=False,
    #     min_importance=0.08,
    #     figsize=(10, 7),
    #     edge_length_scale=0.5,
    #     title='IRF3 Regulatory Network',
    #     show_legend=False
    # )





def compute_global_spatial_correlation_matrix(
    adata,
    layer_key: str = 'counts',
    latent_obsm_key: str = 'spatial',
    batch_key: Optional[str] = None,
    n_neighbors: int = 30,
    model: str = 'bernoulli',
    jobs: int = 4,
    save_path: Optional[str] = None,
    cache: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute global spatial correlation matrix for all genes with batch correction.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    layer_key : str
        Layer containing count data
    latent_obsm_key : str
        Key in adata.obsm for spatial coordinates
    batch_key : str, optional
        Key in adata.obs containing batch labels for batch correction
    n_neighbors : int
        Number of neighbors for spatial correlation computation
    model : str
        Statistical model for Hotspot ('danb', 'bernoulli', 'normal', 'none')
    jobs : int
        Number of parallel jobs
    save_path : str, optional
        Path to save correlation matrices (without extension)
    cache : bool
        Whether to use cached results if available
        
    Returns
    -------
    tuple
        (correlation_matrix, correlation_z_scores)
    """
    
    # Check for cached results
    if cache and save_path:
        corr_path = f"{save_path}_correlation_matrix.csv"
        z_path = f"{save_path}_correlation_z_scores.csv"
        
        if os.path.exists(corr_path) and os.path.exists(z_path):
            print(f"Loading cached correlation matrices from {save_path}")
            correlation_matrix = pd.read_csv(corr_path, index_col=0)
            correlation_z = pd.read_csv(z_path, index_col=0)
            return correlation_matrix, correlation_z
    
    print(f"Computing global spatial correlation matrix for {adata.n_vars} genes...")
    print(f"Using {'batch-corrected' if batch_key else 'standard'} spatial correlation")
    
    # Initialize Hotspot with batch correction
    hs = Hotspot(
        adata,
        layer_key=layer_key,
        model=model,
        latent_obsm_key=latent_obsm_key,
        batch_key=batch_key
    )
    
    # Create KNN graph with batch awareness
    print("Creating spatial neighborhood graph...")
    hs.create_knn_graph(
        weighted_graph=True,
        n_neighbors=n_neighbors,
        batch_aware=batch_key is not None
    )
    
    # Compute pairwise spatial correlations with batch correction
    print("Computing pairwise spatial correlations...")
    
    # Get counts data
    if layer_key and layer_key in adata.layers:
        counts_data = adata.layers[layer_key]
    else:
        counts_data = adata.X
    
    if issparse(counts_data):
        counts_data = counts_data.toarray()
    
    # Create DataFrame for correlation computation
    counts_df = pd.DataFrame(
        counts_data.T,  # Transpose to genes x cells
        index=adata.var_names,
        columns=adata.obs_names
    )
    
    # Get batch information if provided
    batches = adata.obs[batch_key].values if batch_key else None
    
    # Compute correlations using Hotspot's batch-aware method
    try:
        correlation_matrix, correlation_z = compute_hs_pairs_centered_cond(
            counts_df,
            hs.neighbors_numeric if hasattr(hs, 'neighbors_numeric') else hs.neighbors,
            hs.weights,
            hs.umi_counts,
            model,
            jobs=jobs,
            batches=batches
        )
    except Exception as e:
        print(f"Error in batch-corrected correlation computation: {e}")
        print("Falling back to standard correlation computation...")
        correlation_matrix, correlation_z = hs.compute_local_correlations(
            list(adata.var_names), jobs=jobs
        )
    
    print(f"Computed correlation matrix: {correlation_matrix.shape}")
    
    # Save matrices if path provided
    if save_path:
        corr_path = f"{save_path}_correlation_matrix.csv"
        z_path = f"{save_path}_correlation_z_scores.csv"
        correlation_matrix.to_csv(corr_path)
        correlation_z.to_csv(z_path)
        print(f"Saved correlation matrices to {save_path}_*.csv")
    
    return correlation_matrix, correlation_z


def filter_receptors_by_correlation(
    adata,
    tf_name: str,
    correlation_matrix: pd.DataFrame,
    receptor_source: str = 'receptor_dict',
    candidate_receptors: Optional[List[str]] = None,
    target_genes: Optional[List[str]] = None,
    top_percent: float = 0.05,
    min_correlation_threshold: float = 0.0,
    correlation_method: str = 'mean',  # 'mean', 'median', 'max'
    save_results: bool = True,
    output_prefix: str = 'receptor_filtering'
) -> Dict:
    """
    Filter receptors for a given transcription factor based on spatial correlation 
    with downstream target genes.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing GRN analysis results
    tf_name : str
        Name of the transcription factor (e.g., 'IRF4')
    correlation_matrix : pd.DataFrame
        Precomputed spatial correlation matrix
    receptor_source : str
        Key in adata.uns for receptor data
    candidate_receptors : list, optional
        List of candidate receptors to consider. If None, uses all receptors for the TF.
    target_genes : list, optional
        List of target genes to correlate with. If None, uses all regulon targets.
    top_percent : float
        Percentage of top correlated receptors to keep (e.g., 0.05 for top 5%)
    min_correlation_threshold : float
        Minimum correlation threshold to consider
    correlation_method : str
        Method to aggregate correlations ('mean', 'median', 'max')
    save_results : bool
        Whether to save results to files
    output_prefix : str
        Prefix for output files
        
    Returns
    -------
    dict
        Dictionary containing filtered receptors, correlation scores, and metadata
    """
    
    print(f"Filtering receptors for TF: {tf_name}")
    
    # 1. Extract receptors and targets for the TF
    receptor_data = adata.uns.get(receptor_source, {})
    regulon_dict = adata.uns.get('regulon_dict', {})
    
    if tf_name not in receptor_data:
        raise ValueError(f"TF '{tf_name}' not found in {receptor_source}")
    
    tf_key = f"{tf_name}(+)"
    if tf_key not in regulon_dict:
        raise ValueError(f"Regulon '{tf_key}' not found in regulon_dict")
    
    # Get candidate receptors (use provided list or all receptors for TF)
    all_receptors = receptor_data[tf_name]
    if candidate_receptors is None:
        candidate_receptors = all_receptors
    else:
        # Ensure candidate receptors are in the original receptor list
        candidate_receptors = [r for r in candidate_receptors if r in all_receptors]
    
    # Get target genes (use provided list or all regulon targets)
    all_targets = regulon_dict[tf_key]
    if target_genes is None:
        target_genes = all_targets
    else:
        # Ensure target genes are in the original target list
        target_genes = [t for t in target_genes if t in all_targets]
    
    print(f"Analyzing {len(candidate_receptors)} candidate receptors vs {len(target_genes)} target genes")
    
    # 2. Filter to genes present in correlation matrix
    available_receptors = [r for r in candidate_receptors if r in correlation_matrix.index]
    available_targets = [t for t in target_genes if t in correlation_matrix.index]
    
    if not available_receptors:
        raise ValueError("No candidate receptors found in correlation matrix")
    if not available_targets:
        raise ValueError("No target genes found in correlation matrix")
    
    print(f"Found {len(available_receptors)}/{len(candidate_receptors)} receptors and "
          f"{len(available_targets)}/{len(target_genes)} targets in correlation matrix")
    
    # 3. Calculate correlation scores for each receptor
    receptor_scores = {}
    receptor_correlations = {}
    receptor_stats = {}
    
    for receptor in available_receptors:
        # Get correlations between this receptor and all targets
        target_correlations = []
        receptor_target_pairs = {}
        
        for target in available_targets:
            if receptor in correlation_matrix.index and target in correlation_matrix.columns:
                corr_value = correlation_matrix.loc[receptor, target]
                target_correlations.append(corr_value)
                receptor_target_pairs[target] = corr_value
        
        if target_correlations:
            # Calculate aggregate correlation score
            if correlation_method == 'mean':
                agg_correlation = np.mean(target_correlations)
            elif correlation_method == 'median':
                agg_correlation = np.median(target_correlations)
            elif correlation_method == 'max':
                agg_correlation = np.max(target_correlations)
            else:
                raise ValueError(f"Unknown correlation_method: {correlation_method}")
            
            receptor_scores[receptor] = agg_correlation
            receptor_correlations[receptor] = receptor_target_pairs
            receptor_stats[receptor] = {
                'mean': np.mean(target_correlations),
                'median': np.median(target_correlations),
                'std': np.std(target_correlations),
                'min': np.min(target_correlations),
                'max': np.max(target_correlations),
                'n_targets': len(target_correlations)
            }
            
            print(f"  {receptor}: {correlation_method} = {agg_correlation:.4f} "
                  f"(range: {min(target_correlations):.4f} to {max(target_correlations):.4f})")
    
    if not receptor_scores:
        raise ValueError("No valid receptor-target correlations computed")
    
    # 4. Filter receptors based on correlation scores
    # Sort by correlation score (descending)
    sorted_receptors = sorted(receptor_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Apply minimum threshold
    filtered_by_threshold = [(r, s) for r, s in sorted_receptors if s >= min_correlation_threshold]
    
    # Select top percentage
    n_top = max(1, int(len(filtered_by_threshold) * top_percent))
    top_receptors = filtered_by_threshold[:n_top]
    
    print(f"\nFiltering results:")
    print(f"  Total candidate receptors: {len(candidate_receptors)}")
    print(f"  Available in correlation matrix: {len(available_receptors)}")
    print(f"  Above threshold ({min_correlation_threshold}): {len(filtered_by_threshold)}")
    print(f"  Top {top_percent*100}% selected: {len(top_receptors)}")
    
    # 5. Prepare results
    results = {
        'tf_name': tf_name,
        'all_receptors': all_receptors,
        'candidate_receptors': candidate_receptors,
        'available_receptors': available_receptors,
        'filtered_receptors': [r for r, _ in top_receptors],
        'receptor_scores': receptor_scores,
        'receptor_correlations': receptor_correlations,
        'receptor_stats': receptor_stats,
        'top_receptors_with_scores': top_receptors,
        'target_genes': available_targets,
        'filtering_params': {
            'top_percent': top_percent,
            'min_correlation_threshold': min_correlation_threshold,
            'correlation_method': correlation_method,
            'receptor_source': receptor_source
        }
    }
    
    # 6. Save results
    if save_results:
        results_path = f"{output_prefix}_{tf_name}_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary table
        summary_df = pd.DataFrame([
            {
                'receptor': r,
                'score': s,
                **receptor_stats[r]
            }
            for r, s in sorted_receptors
        ])
        summary_path = f"{output_prefix}_{tf_name}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Saved results to {results_path}")
        print(f"Saved summary to {summary_path}")
    
    # Print final results
    print(f"\nTop filtered receptors for {tf_name} (by {correlation_method}):")
    for i, (receptor, score) in enumerate(top_receptors, 1):
        print(f"  {i}. {receptor}: {score:.4f}")
    
    return results


def plot_receptor_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    filtering_results: Dict,
    show_all_receptors: bool = False,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot heatmap of receptor-target correlations.
    
    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Spatial correlation matrix
    filtering_results : dict
        Results from filter_receptors_by_correlation
    show_all_receptors : bool
        If True, show all available receptors. If False, show only filtered ones.
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the plot
    """
    
    tf_name = filtering_results['tf_name']
    
    # Select receptors to show
    if show_all_receptors:
        receptors_to_show = filtering_results['available_receptors']
        title_suffix = "(All Receptors)"
    else:
        receptors_to_show = filtering_results['filtered_receptors']
        title_suffix = "(Filtered Receptors)"
    
    targets_to_show = filtering_results['target_genes']
    
    # Subset correlation matrix
    available_receptors = [r for r in receptors_to_show if r in correlation_matrix.index]
    available_targets = [t for t in targets_to_show if t in correlation_matrix.columns]
    
    if not available_receptors or not available_targets:
        print("No valid receptors or targets for plotting")
        return
    
    subset_matrix = correlation_matrix.loc[available_receptors, available_targets]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Create annotation matrix for filtered receptors
    annot_matrix = None
    if not show_all_receptors and subset_matrix.size <= 50:
        annot_matrix = subset_matrix.round(3)
    
    sns.heatmap(
        subset_matrix,
        cmap='RdBu_r',
        center=0,
        annot=annot_matrix,
        fmt='g' if annot_matrix is not None else None,
        cbar_kws={'label': 'Spatial Correlation'},
        xticklabels=True,
        yticklabels=True
    )
    
    plt.title(f'{tf_name} Receptor-Target Spatial Correlations {title_suffix}')
    plt.xlabel('Target Genes')
    plt.ylabel('Receptor Genes')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_receptor_score_distribution(
    filtering_results: Dict,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of receptor correlation scores.
    
    Parameters
    ----------
    filtering_results : dict
        Results from filter_receptors_by_correlation
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the plot
    """
    
    tf_name = filtering_results['tf_name']
    receptor_scores = filtering_results['receptor_scores']
    filtered_receptors = set(filtering_results['filtered_receptors'])
    
    # Prepare data
    receptors = list(receptor_scores.keys())
    scores = list(receptor_scores.values())
    colors = ['red' if r in filtered_receptors else 'blue' for r in receptors]
    
    plt.figure(figsize=figsize)
    
    # Bar plot
    bars = plt.bar(range(len(receptors)), scores, color=colors, alpha=0.7)
    
    # Add threshold line if applicable
    threshold = filtering_results['filtering_params']['min_correlation_threshold']
    if threshold > 0:
        plt.axhline(y=threshold, color='green', linestyle='--', alpha=0.7, 
                   label=f'Threshold ({threshold})')
    
    plt.xlabel('Receptors')
    plt.ylabel('Correlation Score')
    plt.title(f'{tf_name} Receptor Correlation Scores')
    plt.xticks(range(len(receptors)), receptors, rotation=45, ha='right')
    
    # Add legend
    plt.scatter([], [], color='red', alpha=0.7, label='Filtered (Top)')
    plt.scatter([], [], color='blue', alpha=0.7, label='Not Selected')
    if threshold > 0:
        plt.legend()
    else:
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def update_receptor_dict_with_filtered(
    adata, 
    filtering_results: Dict, 
    target_key: str = 'receptor_dict_filtered'
) -> None:
    """
    Update adata.uns with filtered receptors.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    filtering_results : dict
        Results from filter_receptors_by_correlation
    target_key : str
        Key to store filtered receptors in adata.uns
    """
    
    tf_name = filtering_results['tf_name']
    filtered_receptors = filtering_results['filtered_receptors']
    
    if target_key not in adata.uns:
        adata.uns[target_key] = {}
    
    adata.uns[target_key][tf_name] = filtered_receptors
    
    print(f"Updated adata.uns['{target_key}'] with {len(filtered_receptors)} filtered receptors for {tf_name}")


# Example usage workflow:
"""
# Step 1: Compute global correlation matrix (do this once)
correlation_matrix, correlation_z = compute_global_spatial_correlation_matrix(
    PDAC,
    batch_key='patient',  # Enable batch correction
    save_path='PDAC_spatial_correlations',
    cache=True  # Use cached results if available
)

# Step 2: Extract receptors from GRN plot
fig, G = plot_grn_network(
    PDAC,
    receptor_source='receptor_dict',
    selected_tfs=['IRF4'],
    show_receptor=True,
    min_importance=0.03,
    figsize=(5, 2.5),
    edge_length_scale=0.05,
    title='',
    show_legend=False
)

# Get receptor nodes from the network
all_nodes = list(G.nodes())
print("Network nodes:", all_nodes)

# Step 3: Filter receptors for IRF4
filtering_results = filter_receptors_by_correlation(
    adata=PDAC,
    tf_name='IRF4',
    correlation_matrix=correlation_matrix,
    top_percent=0.05,  # Keep top 5%
    min_correlation_threshold=0.1,
    correlation_method='mean',  # or 'median', 'max'
    save_results=True
)

# Step 4: Visualize results
plot_receptor_correlation_heatmap(
    correlation_matrix, 
    filtering_results,
    show_all_receptors=False,  # Show only filtered receptors
    save_path='IRF4_receptor_heatmap.png'
)

plot_receptor_score_distribution(
    filtering_results,
    save_path='IRF4_receptor_scores.png'
)

# Step 5: Update receptor dictionary and replot network
update_receptor_dict_with_filtered(PDAC, filtering_results)

# Plot network with filtered receptors
fig, G = plot_grn_network(
    PDAC,
    receptor_source='receptor_dict_filtered',  # Use filtered receptors
    selected_tfs=['IRF4'],
    show_receptor=True,
    min_importance=0.03,
    figsize=(5, 2.5),
    edge_length_scale=0.05,
    title='IRF4 Network (Filtered Receptors)',
    show_legend=False
)
"""
