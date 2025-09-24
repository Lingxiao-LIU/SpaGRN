import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import seaborn as sns
from matplotlib.patches import Rectangle
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
