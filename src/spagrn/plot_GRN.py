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
    layout_type: str = 'layered',
    node_size_scale: float = 1.0,
    edge_width_scale: float = 1.0,
    show_edge_labels: bool = False,
    color_scheme: str = 'default',
    figsize: Tuple[int, int] = (7, 7),
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
        Key in adata.uns for receptor data. Options:
        - 'receptor_dict_all': all receptors
        - 'receptor_dict': filtered receptors  
        - 'receptor_dict_diff': difference (receptor_dict_all - receptor_dict)
    min_importance : float, default 0.1
        Minimum edge importance threshold for TF-target connections
    max_nodes : int, optional
        Maximum number of nodes to display (prioritizes by importance)
    selected_tfs : list of str, optional
        Specific TFs to include. If None, includes all TFs
    selected_receptors : list of str, optional
        Specific receptors to include. If None, includes all receptors
    layout_type : str, default 'layered'
        Network layout type: 'layered', 'spring', 'circular', 'hierarchical'
    node_size_scale : float, default 1.0
        Scaling factor for node sizes
    edge_width_scale : float, default 1.0
        Scaling factor for edge widths
    show_edge_labels : bool, default False
        Whether to show importance values on edges
    color_scheme : str, default 'default'
        Color scheme: 'default', 'pastel', 'bright', 'earth'
    figsize : tuple, default (15, 10)
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot
    title : str, optional
        Plot title
    show_legend : bool, default True
        Whether to show legend
    font_size : int, default 8
        Font size for labels
    dpi : int, default 300
        DPI for saved figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    G : networkx.DiGraph
        The network graph object
        
    Examples
    --------
    # Basic usage
    fig, G = plot_grn_network(adata)
    
    # Use all receptors with custom filtering
    fig, G = plot_grn_network(
        adata, 
        receptor_source='receptor_dict_all',
        min_importance=0.2,
        max_nodes=50
    )
    
    # Focus on specific TFs
    fig, G = plot_grn_network(
        adata,
        selected_tfs=['TF1', 'TF2'],
        layout_type='spring',
        color_scheme='earth'
    )
    """
    
    # Validate inputs
    if receptor_source not in ['receptor_dict', 'receptor_dict_all', 'receptor_dict_diff']:
        raise ValueError("receptor_source must be 'receptor_dict', 'receptor_dict_all', or 'receptor_dict_diff'")
    
    # Extract data from adata
    try:
        if receptor_source == 'receptor_dict_diff':
            receptor_dict_all = adata.uns.get('receptor_dict_all', {})
            receptor_dict = adata.uns.get('receptor_dict', {})
            # Calculate difference
            receptor_data = {}
            for tf, receptors_all in receptor_dict_all.items():
                receptors_filtered = receptor_dict.get(tf, [])
                diff_receptors = list(set(receptors_all) - set(receptors_filtered))
                if diff_receptors:
                    receptor_data[tf] = diff_receptors
        else:
            receptor_data = adata.uns.get(receptor_source, {})
            
        regulon_dict = adata.uns.get('regulon_dict', {})
        adj_data = adata.uns.get('adj', [])
        
        if isinstance(adj_data, pd.DataFrame):
            adj_data = adj_data.to_dict('records')
            
    except KeyError as e:
        raise KeyError(f"Required data not found in adata.uns: {e}")
    
    # Validate data availability
    if not receptor_data:
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
    
    # Add receptor-TF connections
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
            if tf in tf_nodes:  # Only add if TF is already in our network
                target_nodes.add(target)
                G.add_edge(tf, target, edge_type='tf_target', weight=importance)
                tf_target_edges[(tf, target)] = importance
    
    # Limit nodes if specified
    if max_nodes and len(G.nodes()) > max_nodes:
        # Prioritize by node degree and edge importance
        node_scores = {}
        for node in G.nodes():
            if node in tf_nodes:
                # TFs get high priority
                score = G.degree(node) * 10
            elif node in receptor_nodes:
                score = G.degree(node) * 5
            else:  # target nodes
                # Score targets by sum of incoming edge weights
                score = sum(G[pred][node]['weight'] for pred in G.predecessors(node))
            node_scores[node] = score
        
        # Keep top nodes
        top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        keep_nodes = [node for node, _ in top_nodes]
        G = G.subgraph(keep_nodes).copy()
        
        # Update node sets
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
    
    # Calculate layout
    if layout_type == 'layered':
        pos = _calculate_layered_layout(G, receptor_nodes, tf_nodes, target_nodes)
    elif layout_type == 'spring':
        pos = nx.spring_layout(G, k=3, iterations=50)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'hierarchical':
        pos = nx.shell_layout(G, nlist=[receptor_nodes, tf_nodes, target_nodes])
    else:
        pos = nx.spring_layout(G)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Draw edges
    receptor_tf_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'receptor_tf']
    tf_target_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'tf_target']
    
    # Draw receptor-TF edges
    if receptor_tf_edges:
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
    
    # Draw nodes
    node_sizes = {
        'receptor': 300 * node_size_scale,
        'tf': 500 * node_size_scale,
        'target': 200 * node_size_scale
    }
    
    # Draw each node type separately
    for node_type, nodes, color in [
        ('receptor', receptor_nodes, colors['receptor']),
        ('tf', tf_nodes, colors['tf']),
        ('target', target_nodes, colors['target'])
    ]:
        if nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes,
                node_color=color, node_size=node_sizes[node_type],
                alpha=0.8, ax=ax
            )
    
    # Draw labels
    label_pos = {node: (x, y+0.05) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=font_size, ax=ax)
    
    # Draw edge labels if requested
    if show_edge_labels:
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            if d['edge_type'] == 'tf_target':
                edge_labels[(u, v)] = f"{d['weight']:.2f}"
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=font_size-2, ax=ax)
    
    # Add legend
    if show_legend:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['receptor'], 
                      markersize=10, label=f'Receptors (n={len(receptor_nodes)})'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['tf'], 
                      markersize=12, label=f'TFs (n={len(tf_nodes)})'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['target'], 
                      markersize=8, label=f'Targets (n={len(target_nodes)})'),
            plt.Line2D([0], [0], color=colors['receptor_tf_edge'], linewidth=2, 
                      label='Receptor→TF'),
            plt.Line2D([0], [0], color=colors['tf_target_edge'], linewidth=2, 
                      label='TF→Target')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.2))
    
    # Set title
    if title is None:
        title = f'Gene Regulatory Network ({receptor_source.replace("_", " ").title()})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add network statistics as text
    stats_text = f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())}\n"
    stats_text += f"Min importance: {min_importance}"
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    # Remove axes
    ax.axis('off')
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")
    
    return fig, G


def _calculate_layered_layout(G, receptor_nodes, tf_nodes, target_nodes):
    """Calculate layered layout positions for GRN visualization."""
    pos = {}
    
    # Layer positions
    layers = {
        'receptor': {'x': 0, 'y_center': 0},
        'tf': {'x': 1, 'y_center': 0}, 
        'target': {'x': 2, 'y_center': 0}
    }
    
    # Position receptors
    receptor_list = list(receptor_nodes)
    if receptor_list:
        y_spacing = 1.0 if len(receptor_list) <= 1 else 2.0 / (len(receptor_list) - 1)
        for i, receptor in enumerate(receptor_list):
            y = -1.0 + i * y_spacing if len(receptor_list) > 1 else 0
            pos[receptor] = (layers['receptor']['x'], y)
    
    # Position TFs
    tf_list = list(tf_nodes)
    if tf_list:
        y_spacing = 1.0 if len(tf_list) <= 1 else 2.0 / (len(tf_list) - 1)
        for i, tf in enumerate(tf_list):
            y = -1.0 + i * y_spacing if len(tf_list) > 1 else 0
            pos[tf] = (layers['tf']['x'], y)
    
    # Position targets
    target_list = list(target_nodes)
    if target_list:
        y_spacing = 0.8 if len(target_list) <= 1 else 1.6 / (len(target_list) - 1)
        for i, target in enumerate(target_list):
            y = -0.8 + i * y_spacing if len(target_list) > 1 else 0
            pos[target] = (layers['target']['x'], y)
    
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
        if receptor_source == 'receptor_dict_diff':
            receptor_dict_all = adata.uns.get('receptor_dict_all', {})
            receptor_dict = adata.uns.get('receptor_dict', {})
            receptor_data = {}
            for tf, receptors_all in receptor_dict_all.items():
                receptors_filtered = receptor_dict.get(tf, [])
                diff_receptors = list(set(receptors_all) - set(receptors_filtered))
                if diff_receptors:
                    receptor_data[tf] = diff_receptors
        else:
            receptor_data = adata.uns.get(receptor_source, {})
            
        regulon_dict = adata.uns.get('regulon_dict', {})
        adj_data = adata.uns.get('adj', [])
        
        if isinstance(adj_data, pd.DataFrame):
            adj_data = adj_data.to_dict('records')
    except KeyError:
        return {}
    
    # Calculate statistics
    total_receptors = len(set(receptor for receptors in receptor_data.values() for receptor in receptors))
    total_tfs = len(receptor_data)
    total_targets = len(set(target for targets in regulon_dict.values() for target in targets))
    total_edges = len(adj_data)
    
    # Calculate importance statistics
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


# Example usage and testing function
def test_grn_plot():
    """Test function with sample data."""
    import anndata as ad
    
    # Create sample AnnData object
    sample_data = {
        'receptor_dict_all': {
            'TF1': ['EGFR', 'PDGFRA', 'FGFR1'],
            'TF2': ['VEGFR2', 'NOTCH1', 'WNT3A'],
            'TF3': ['TGFβR1', 'BMPR1A']
        },
        'receptor_dict': {
            'TF1': ['EGFR', 'PDGFRA'],
            'TF2': ['VEGFR2', 'NOTCH1'],
            'TF3': ['TGFβR1']
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
    
    # Create dummy AnnData object
    adata = ad.AnnData(np.random.rand(100, 20))
    adata.uns.update(sample_data)
    
    # Test different receptor sources
    for source in ['receptor_dict', 'receptor_dict_all', 'receptor_dict_diff']:
        print(f"\nTesting with receptor_source='{source}'")
        fig, G = plot_grn_network(adata, receptor_source=source, title=f"GRN Network ({source})")
        if fig:
            plt.show()
        
        # Print summary
        summary = get_network_summary(adata, source)
        print("Network Summary:", summary)

if __name__ == "__main__":
    test_grn_plot()
    
    
    # Basic usage
    fig, G = plot_grn_network(adata)

    # Use all receptors with filtering
    fig, G = plot_grn_network(
        adata, 
        receptor_source='receptor_dict_all',
        min_importance=0.2,
        max_nodes=50
    )

    # Focus on specific TFs with earth styling
    fig, G = plot_grn_network(
        adata,
        selected_tfs=['TF1', 'TF2'],
        color_scheme='earth',
        show_edge_labels=True,
    )

    # Show difference between all and filtered receptors
    fig, G = plot_grn_network(
        adata,
        receptor_source='receptor_dict_diff',
        title='Filtered Out Receptors'
    )