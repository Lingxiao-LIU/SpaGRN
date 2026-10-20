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
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
    font_size: int = 12,
    dpi: int = 300,
    filter_receptor: bool = False,
    correlation_matrix: Optional[pd.DataFrame] = None,
    correlation_z: Optional[pd.DataFrame] = None,
    top_percent: Optional[float] = None,
    min_correlation_threshold: float = 0.05,
    min_z_threshold: float = 5.5,
    top_n: Optional[int] = None,
    correlation_method: str = 'mean',
    not_potential_target: bool = False,
    show_heatmap: bool = False,
    show_score_dist: bool = False,
    figsize: Tuple[int, int] = (12, 8),      
    score_dist_figsize: Tuple[int, int] = (10, 6),   
    label_side: str = 'center',
    legend_loc: str = 'best',
) -> Tuple[plt.Figure, nx.DiGraph]:
    """
    Create a Gene Regulatory Network plot showing receptors -> TFs -> target genes.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing GRN analysis results
    receptor_source : str, default 'receptor_dict'
        Key in adata.uns for receptor data. Only 'receptor_dict' is supported.
    min_importance : float or str, default 0.1
        Minimum edge importance threshold for TF-target connections. May also be
        'auto' (75th percentile of the adjacency importances for the selected TFs)
        or 'pNN' for an explicit percentile (e.g. 'p90'). Use this when the
        'importance' column is correlation-like rather than GRNBoost-scale, where
        the 0.1 default would drop every target edge.
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
    filter_receptor : bool, default False
        Whether to filter receptors using spatial correlation
    correlation_matrix : pd.DataFrame, optional
        Precomputed spatial correlation matrix (required if filter_receptor=True)
    correlation_z : pd.DataFrame, optional
        Precomputed spatial correlation z-scores (required if filter_receptor=True)
    top_percent : float, optional
        Percentage of top correlated receptors to keep
    min_correlation_threshold : float, default 0.05
        Minimum correlation threshold
    min_z_threshold : float, default 5.5
        Minimum z-score threshold
    top_n : int, optional
        Select top N receptors, overriding other filters
    correlation_method : str, default 'mean'
        Method to aggregate correlations ('mean', 'median', 'max')
    show_heatmap : bool, default False
        Whether to show receptor-target correlation heatmap
    show_score_dist : bool, default False
        Whether to show receptor score distribution plot
    score_dist_figsize : tuple, default (10, 6)
        Figure size for score distribution plot (width, height)
    label_side : {'center', 'outside'}, default 'center'
        'center' writes gene names on the nodes. 'outside' writes receptor names to the
        left of their nodes, target names to the right and TF names above, so small
        print-size nodes stay readable.
    legend_loc : {'best', 'below'}, default 'best'
        'below' puts a frameless legend under the network (saves width in print figures).

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

    if filter_receptor:
        if correlation_matrix is None or correlation_z is None:
            raise ValueError("correlation_matrix and correlation_z are required when filter_receptor=True")
        if not show_receptor:
            warnings.warn("filter_receptor=True but show_receptor=False; filtering may not affect the plot")

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
    else:
        selected_tfs = list(receptor_data.keys())

    # Resolve a string/'auto'/'pNN' min_importance to a numeric threshold.
    # Useful when 'importance' is correlation-like (small/signed) rather than
    # GRNBoost-scale, where the hardcoded 0.1 default drops every target edge.
    if isinstance(min_importance, str):
        rel = [e.get('importance', 0) for e in adj_data if e.get('TF', '') in selected_tfs]
        if not rel:
            rel = [e.get('importance', 0) for e in adj_data]
        rel = np.asarray(rel, dtype=float)
        token = min_importance.lower()
        if token == 'auto':
            pct = 75.0
        elif token.startswith('p') and token[1:].replace('.', '', 1).isdigit():
            pct = float(token[1:])
        else:
            raise ValueError(
                "min_importance must be a float, 'auto', or 'pNN' (e.g. 'p90')")
        min_importance = float(np.percentile(rel, pct)) if rel.size else 0.0
        print(f"min_importance='{token}' resolved to {min_importance:.4g} "
              f"({pct:.0f}th percentile of {rel.size} adj importances for selected TFs)")

    # Filter adjacency data by importance and selected TFs
    filtered_adj = []
    for edge in adj_data:
        if edge.get('importance', 0) >= min_importance:
            tf_name = edge.get('TF', '')
            if tf_name in selected_tfs:
                filtered_adj.append(edge)

    # Collect targets per TF based on filtered_adj
    tf_to_targets = {}
    for edge in filtered_adj:
        tf = edge['TF']
        if tf not in tf_to_targets:
            tf_to_targets[tf] = []
        tf_to_targets[tf].append(edge['target'])

    # Filter receptors if requested
    filtered_receptor_data = {}
    filtering_results = {}
    if filter_receptor and show_receptor:
        top_percent_val = top_percent if top_percent is not None else 0.05

        # When restricting to potential UPSTREAM receptors (not downstream targets),
        # draw candidates from the pre-pruning potential_receptor_dict.
        potential_dict = adata.uns.get('potential_receptor_dict', {}) if not_potential_target else {}
        if not_potential_target and not potential_dict:
            warnings.warn(
                "not_potential_target=True but adata.uns['potential_receptor_dict'] is missing. "
                "Run compute_receptor_regulon_correlations (or infer() with receptor_corr=True) first. "
                "Falling back to receptor_dict."
            )

        tf_candidate_pool = potential_dict if (not_potential_target and potential_dict) else receptor_data

        for tf in selected_tfs:
            if tf not in tf_candidate_pool:
                continue
            target_genes = tf_to_targets.get(tf, None)
            # candidate receptors for this TF
            cand = selected_receptors
            if not_potential_target and potential_dict:
                cand = potential_dict.get(tf, [])
                if selected_receptors:
                    cand = [r for r in cand if r in selected_receptors]
            results = filter_receptors_by_correlation(
                adata=adata,
                tf_name=tf,
                correlation_matrix=correlation_matrix,
                correlation_z=correlation_z,
                receptor_source=receptor_source,
                candidate_receptors=cand,
                target_genes=target_genes,
                top_percent=top_percent_val,
                min_correlation_threshold=min_correlation_threshold,
                min_z_threshold=min_z_threshold,
                top_n=top_n,
                correlation_method=correlation_method,
                save_results=False
            )
            filtered = list(results['filtered_receptors'])

            # Drop any receptor that is itself a pruned downstream target of this TF.
            if not_potential_target:
                pruned_targets = set(regulon_dict.get(f"{tf}(+)", [])) | set(regulon_dict.get(tf, []))
                filtered = [r for r in filtered if r not in pruned_targets]

            filtering_results[tf] = results
            filtered_receptor_data[tf] = filtered

        if filtered_receptor_data:
            key = 'receptor_dict_upstream' if not_potential_target else 'receptor_dict_filtered'
            adata.uns[key] = filtered_receptor_data
            receptor_data = filtered_receptor_data

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

    # Warn loudly when the target layer is empty (a common, silent failure):
    # almost always min_importance is set above the adj importance range.
    if not target_nodes:
        imps = [e.get('importance', 0) for e in adj_data]
        mx = max(imps) if imps else float('nan')
        warnings.warn(
            f"0 TF->target edges passed min_importance={min_importance:g} "
            f"(adj importance ranges up to {mx:.3g}). The plot will have no "
            f"target genes. Lower the threshold (e.g. min_importance=0.0, "
            f"'auto', or 'p75'), or check that adj['TF'] names match "
            f"regulon_dict keys."
        )

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

    if label_side == 'outside':        # one evenly spaced column per node type, using the full height
        columns = [(receptor_nodes if show_receptor else set(), 0.1), (tf_nodes, 0.45),
                   (set(pos) - set(tf_nodes) - (set(receptor_nodes) if show_receptor else set()), 0.8)]
        for nodes, x in columns:
            nodes = sorted((n for n in nodes if n in pos), key=lambda n: -pos[n][1])
            for k, n in enumerate(nodes):
                pos[n] = (x, 0.5 if len(nodes) == 1 else 0.95 - 0.9 * k / (len(nodes) - 1))

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
            alpha=0.7, ax=ax, arrows=True, arrowsize=20 * edge_width_scale
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
                width=width, alpha=alpha, ax=ax, arrows=True, arrowsize=15 * edge_width_scale
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
    if label_side == 'outside':        # offsets in points, so they hold at any figure size
        for node, (x, y) in pos.items():
            if node in tf_nodes:
                xy, ha, va = (0, 7), 'center', 'bottom'
            elif show_receptor and node in receptor_nodes:
                xy, ha, va = (-7, 0), 'right', 'center'
            else:
                xy, ha, va = (6, 0), 'left', 'center'
            # white backing keeps names readable where edges pass behind them
            ax.annotate(node, (x, y), xytext=xy, textcoords='offset points', ha=ha, va=va, fontsize=font_size,
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85), zorder=5)
    else:
        nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)

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
        if legend_loc == 'below':
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.0),
                      ncol=3, frameon=False, fontsize=font_size)
        else:
            ax.legend(handles=legend_elements, loc='best', fontsize=font_size, bbox_to_anchor=(1.0, 1.0))

    # Set title (if provided)
    if title is None:
        title = f'Gene Regulatory Network ({"TFs to Targets" if not show_receptor else receptor_source.replace("_", " ").title()})'
    if title:
        ax.set_title(title, fontsize=18, fontweight='bold')

    # Adjust plot margins to prevent clipping
    x_coords = [x for x, y in pos.values()]
    y_coords = [y for x, y in pos.values()]
    if x_coords and y_coords:
        x_margin = 0.3 * (max(x_coords) - min(x_coords)) + 0.1
        y_margin = 0.3 * (max(y_coords) - min(y_coords)) + 0.1
        if label_side == 'outside':    # columns already fill the height; keep room only for labels
            x_margin, y_margin = 0.12, 0.04
        ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
        ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)

    # Remove axes
    ax.axis('off')
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")

    plt.show()

    if filter_receptor:
        for tf, results in filtering_results.items():
            if show_heatmap:
                plot_receptor_correlation_heatmap(
                    correlation_matrix=correlation_matrix,
                    filtering_results=results,
                    show_all_receptors=False,
                    figsize=figsize,  # CHANGE THIS LINE
                    save_path=None
                )

            if show_score_dist:
                plot_receptor_score_distribution(
                    filtering_results=results,
                    figsize=score_dist_figsize,  # CHANGE THIS LINE
                    save_path=None
                )

    # Print categorized nodes
    receptor_list = list(receptor_nodes)
    tf_list = list(tf_nodes)
    target_list = list(target_nodes)
    print(f"Receptor: {receptor_list}. \nTF: {tf_list}, \nTarget: {target_list}")

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


def export_grn_network(G, path, fmt='graphml', node_types=None):
    """Serialize a GRN graph (from ``plot_grn_network``) for Cytoscape/Gephi/etc.

    ``plot_grn_network`` returns an in-memory ``networkx.DiGraph`` but nothing
    persists it. This writes the receptor -> TF -> target network to a standard
    interchange format so it can be reopened in external network tools or
    re-loaded for downstream analysis.

    Parameters
    ----------
    G : networkx.DiGraph
        The graph returned by ``plot_grn_network`` (edges carry ``edge_type``
        and ``weight`` attributes).
    path : str
        Output file path. The extension is not required to match ``fmt``.
    fmt : {'graphml', 'gml', 'edgelist', 'cytoscape'}, default 'graphml'
        - 'graphml'/'gml': node/edge attributes preserved (best for Cytoscape/Gephi).
        - 'edgelist': a CSV with columns source,target,edge_type,weight.
        - 'cytoscape': Cytoscape.js-style JSON.
    node_types : dict or None
        Optional mapping ``{node_name: 'receptor'|'tf'|'target'}`` written onto
        each node as a ``node_type`` attribute. Useful because the categories are
        computed during plotting but not stored on the graph.

    Returns
    -------
    str
        The path written.
    """
    import json

    if G is None or G.number_of_nodes() == 0:
        raise ValueError("Graph is empty; nothing to export.")

    # Work on a copy so we don't mutate the caller's graph
    H = G.copy()

    if node_types:
        nx.set_node_attributes(H, {n: node_types.get(n, 'unknown') for n in H.nodes()},
                               name='node_type')

    fmt = fmt.lower()
    if fmt in ('graphml', 'gml'):
        # GraphML/GML cannot serialize None attribute values
        for _, _, d in H.edges(data=True):
            for k, v in list(d.items()):
                if v is None:
                    d[k] = ''
        for _, d in H.nodes(data=True):
            for k, v in list(d.items()):
                if v is None:
                    d[k] = ''
        if fmt == 'graphml':
            nx.write_graphml(H, path)
        else:
            nx.write_gml(H, path)
    elif fmt == 'edgelist':
        rows = [{'source': u, 'target': v,
                 'edge_type': d.get('edge_type', ''),
                 'weight': d.get('weight', '')}
                for u, v, d in H.edges(data=True)]
        pd.DataFrame(rows, columns=['source', 'target', 'edge_type', 'weight']).to_csv(
            path, index=False)
    elif fmt == 'cytoscape':
        with open(path, 'w') as f:
            json.dump(nx.cytoscape_data(H), f, indent=2)
    else:
        raise ValueError(
            f"Unsupported fmt '{fmt}'. Use 'graphml', 'gml', 'edgelist', or 'cytoscape'.")

    print(f"Exported GRN network ({H.number_of_nodes()} nodes, "
          f"{H.number_of_edges()} edges) to {path} [{fmt}]")
    return path

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
    cache: bool = True,
    genes: Optional[List[str]] = None,
    hs=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute a batch-corrected pairwise spatial correlation matrix.

    By default computes over ALL genes (the original behaviour). Pass `genes` to
    restrict the computation to a subset (rows AND columns) — strongly recommended,
    since the all-genes matrix is O(n_genes^2) pairs and is prohibitively slow on
    large panels. Pass `hs` to reuse an already-built (batch-aware) Hotspot graph
    instead of rebuilding it.
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
    
    # Resolve the gene set to correlate (subset strongly recommended)
    if genes is not None:
        genes = [g for g in dict.fromkeys(genes) if g in set(adata.var_names)]
        if len(genes) < 2:
            raise ValueError("`genes` must contain at least 2 genes present in adata.var_names.")
    n_compute = adata.n_vars if genes is None else len(genes)
    print(f"Computing spatial correlation matrix for {n_compute} genes "
          f"({'subset' if genes is not None else 'ALL genes'})...")
    print(f"Using {'batch-corrected' if batch_key else 'standard'} spatial correlation")

    # Initialize / reuse Hotspot with batch correction
    if hs is None:
        hs = Hotspot(
            adata,
            layer_key=layer_key,
            model=model,
            latent_obsm_key=latent_obsm_key,
            batch_key=batch_key
        )
        # Create KNN graph with batch awareness (no cross-sample edges)
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

    # Create DataFrame for correlation computation (genes x cells), subset if asked
    counts_df = pd.DataFrame(
        np.asarray(counts_data).T,  # Transpose to genes x cells
        index=adata.var_names,
        columns=adata.obs_names
    )
    if genes is not None:
        counts_df = counts_df.loc[genes]

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
    correlation_z: pd.DataFrame,
    receptor_source: str = 'receptor_dict',
    candidate_receptors: Optional[List[str]] = None,
    target_genes: Optional[List[str]] = None,
    top_percent: float = 0.05,
    min_correlation_threshold: float = 0.0,
    min_z_threshold: float = 0.0,
    top_n: Optional[int] = None,
    correlation_method: str = 'mean',  # 'mean', 'median', 'max'
    save_results: bool = False,
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
    correlation_z : pd.DataFrame
        Precomputed spatial correlation z-scores matrix
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
    min_z_threshold : float
        Minimum average z-score threshold for significance
    top_n : int, optional
        If provided, select the top N receptors by correlation score, overriding thresholds and top_percent.
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
    
    # Candidate receptors: use the explicitly provided list when given (e.g. the
    # pre-pruning potential receptors); otherwise fall back to all receptors the
    # original receptor_dict assigned to this TF.
    original_receptor_data = adata.uns.get('receptor_dict', {})
    all_receptors = original_receptor_data.get(tf_name, [])

    if candidate_receptors is None:
        candidate_receptors = list(all_receptors)
    else:
        candidate_receptors = list(candidate_receptors)

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
    
    # 3. Calculate correlation scores and z-scores for each receptor
    receptor_scores = {}
    receptor_z_scores = {}
    receptor_correlations = {}
    receptor_z_correlations = {}
    receptor_stats = {}
    
    for receptor in available_receptors:
        # Get correlations and z-scores between this receptor and all targets
        target_correlations = []
        target_z_scores = []
        receptor_target_pairs = {}
        receptor_target_z_pairs = {}
        
        for target in available_targets:
            if receptor in correlation_matrix.index and target in correlation_matrix.columns:
                corr_value = correlation_matrix.loc[receptor, target]
                target_correlations.append(corr_value)
                receptor_target_pairs[target] = corr_value
                
                z_value = correlation_z.loc[receptor, target]
                target_z_scores.append(z_value)
                receptor_target_z_pairs[target] = z_value
        
        if target_correlations and target_z_scores:
            # Calculate aggregate correlation score
            if correlation_method == 'mean':
                agg_correlation = np.mean(target_correlations)
                agg_z = np.mean(target_z_scores)
            elif correlation_method == 'median':
                agg_correlation = np.median(target_correlations)
                agg_z = np.median(target_z_scores)
            elif correlation_method == 'max':
                agg_correlation = np.max(target_correlations)
                agg_z = np.max(target_z_scores)
            else:
                raise ValueError(f"Unknown correlation_method: {correlation_method}")
            
            receptor_scores[receptor] = agg_correlation
            receptor_z_scores[receptor] = agg_z
            receptor_correlations[receptor] = receptor_target_pairs
            receptor_z_correlations[receptor] = receptor_target_z_pairs
            receptor_stats[receptor] = {
                'mean_corr': np.mean(target_correlations),
                'median_corr': np.median(target_correlations),
                'std_corr': np.std(target_correlations),
                'min_corr': np.min(target_correlations),
                'max_corr': np.max(target_correlations),
                'mean_z': np.mean(target_z_scores),
                'median_z': np.median(target_z_scores),
                'std_z': np.std(target_z_scores),
                'min_z': np.min(target_z_scores),
                'max_z': np.max(target_z_scores),
                'n_targets': len(target_correlations)
            }
            
            print(f"  {receptor}: {correlation_method}_corr = {agg_correlation:.4f} "
                  f"(range: {min(target_correlations):.4f} to {max(target_correlations):.4f}), "
                  f"{correlation_method}_z = {agg_z:.4f} "
                  f"(range: {min(target_z_scores):.4f} to {max(target_z_scores):.4f})")
    
    if not receptor_scores:
        raise ValueError("No valid receptor-target correlations computed")
    
    # 4. Filter receptors based on correlation scores and z-scores
    # Sort by correlation score (descending)
    sorted_receptors = sorted(receptor_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate receptors above thresholds
    filtered_by_threshold = [
        (r, s) for r, s in sorted_receptors 
        if s >= min_correlation_threshold and receptor_z_scores[r] >= min_z_threshold
    ]
    num_above_threshold = len(filtered_by_threshold)
    
    print(f"\nFiltering results:")
    print(f"  Total candidate receptors: {len(candidate_receptors)}")
    print(f"  Available in correlation matrix: {len(available_receptors)}")
    print(f"  Above thresholds (corr >= {min_correlation_threshold}, z >= {min_z_threshold}): {num_above_threshold}")
    
    # Apply top_n or top_percent
    if top_n is not None:
        n_top = min(top_n, len(sorted_receptors))
        top_receptors = sorted_receptors[:n_top]
        print(f"  Selected top {top_n} (overriding thresholds and percentage): {len(top_receptors)}")
    else:
        n_top = max(1, int(num_above_threshold * top_percent))
        top_receptors = filtered_by_threshold[:n_top]
        print(f"  Top {top_percent*100}% selected: {len(top_receptors)}")
    
    # 5. Prepare results
    results = {
        'tf_name': tf_name,
        'all_receptors': all_receptors,  # This will now always be from original receptor_dict
        'candidate_receptors': candidate_receptors,
        'available_receptors': available_receptors,
        'filtered_receptors': [r for r, _ in top_receptors],
        'receptor_scores': receptor_scores,
        'receptor_z_scores': receptor_z_scores,
        'receptor_correlations': receptor_correlations,
        'receptor_z_correlations': receptor_z_correlations,
        'receptor_stats': receptor_stats,
        'top_receptors_with_scores': top_receptors,
        'target_genes': available_targets,
        'filtering_params': {
            'top_percent': top_percent,
            'min_correlation_threshold': min_correlation_threshold,
            'min_z_threshold': min_z_threshold,
            'top_n': top_n,
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
                'z_score': receptor_z_scores[r],
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
        z_score = receptor_z_scores[receptor]
        min_corr = receptor_stats[receptor]['min_corr']
        max_corr = receptor_stats[receptor]['max_corr']
        min_z = receptor_stats[receptor]['min_z']
        max_z = receptor_stats[receptor]['max_z']
        print(f"  {i}. {receptor}: {correlation_method}_corr = {score:.4f} "
              f"(range: {min_corr:.4f} to {max_corr:.4f}), "
              f"{correlation_method}_z = {z_score:.4f} "
              f"(range: {min_z:.4f} to {max_z:.4f})")
    
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
    
    # Create plot with custom axes for heatmap and colorbar
    fig, (ax, cax) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 0.05], 'wspace': 0.3})
    
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
        ax=ax,
        cbar_ax=cax
    )
    
    ax.set_title(f'{tf_name} Receptor-Target Spatial Correlations {title_suffix}')
    ax.set_xlabel('Target Genes')
    ax.set_ylabel('Receptor Genes')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
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




# Example usage workflow:
"""
# Step 1: Compute global correlation matrix (do this once)
correlation_matrix, correlation_z = compute_global_spatial_correlation_matrix(
    PDAC,
    batch_key='patient',  # Enable batch correction
    save_path='PDAC_spatial_correlations',
    cache=True  # Use cached results if available
)

#IRF4 centered GRN network plot with receptor filtering
fig, G = plot_grn_network(
    PDAC,
    receptor_source='receptor_dict',
    selected_tfs=['IRF4'],
    show_receptor=True,  # Exclude receptor nodes and edges
    min_importance=0.02, # Target genes with importance score >= 
    filter_receptor=True,
    correlation_matrix=correlation_matrix,
    correlation_z=correlation_z,
    top_n=2, # Overide
    top_percent=1,  # Keep top ratio
    min_correlation_threshold=0.05,
    min_z_threshold= 5.5, # p<0.00000005
    figsize=(5, 3),
    title='', # No title
    show_legend=False,
    show_heatmap=True,
    show_score_dist=True,
    figsize=(7, 4),     
    score_dist_figsize=(7, 4)    
)
"""



"""
Solution for filtering common receptors across multiple regulons in plot_grn_network()

This solution provides two approaches:
1. Add a new parameter to filter for common receptors
2. Use a helper function to pre-filter receptors before calling plot_grn_network
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set


def filter_common_receptors_multi_regulon(
    adata,
    selected_tfs: List[str],
    correlation_matrix: pd.DataFrame,
    correlation_z: pd.DataFrame,
    receptor_source: str = 'receptor_dict',
    min_correlation_threshold: float = 0.05,
    min_z_threshold: float = 5.5,
    top_n_per_tf: Optional[int] = None,
    top_percent_per_tf: Optional[float] = 0.05,
    correlation_method: str = 'mean',
    commonality_threshold: int = None,  # If None, requires receptors common to ALL TFs
    return_union: bool = False  # If True, returns union instead of intersection
) -> Dict[str, List[str]]:
    """
    Filter receptors that are common (or union) across multiple regulons.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing GRN analysis results
    selected_tfs : list of str
        List of TF names to analyze
    correlation_matrix : pd.DataFrame
        Spatial correlation matrix
    correlation_z : pd.DataFrame
        Spatial correlation z-scores
    receptor_source : str
        Key in adata.uns for receptor data
    min_correlation_threshold : float
        Minimum correlation threshold for filtering
    min_z_threshold : float
        Minimum z-score threshold for filtering
    top_n_per_tf : int, optional
        Number of top receptors to consider per TF
    top_percent_per_tf : float, optional
        Percentage of top receptors to consider per TF
    correlation_method : str
        Method to aggregate correlations ('mean', 'median', 'max')
    commonality_threshold : int, optional
        Minimum number of TFs a receptor must appear in.
        If None, requires appearance in ALL TFs (strict intersection).
        Example: If you have 3 TFs and set commonality_threshold=2, 
        receptors appearing in at least 2 TFs will be kept.
    return_union : bool
        If True, return union of all filtered receptors instead of intersection
        
    Returns
    -------
    dict
        Dictionary mapping each TF to its filtered common receptors
        
    Example
    -------
    # Find receptors common to all three regulons
    filtered_receptor_dict = filter_common_receptors_multi_regulon(
        PDAC,
        selected_tfs=['SOX8', 'RXRA', 'SMAD2'],
        correlation_matrix=correlation_matrix,
        correlation_z=correlation_z,
        top_n_per_tf=10,  # Consider top 10 receptors per TF
        commonality_threshold=None  # Must appear in all 3
    )
    
    # Find receptors appearing in at least 2 out of 3 regulons
    filtered_receptor_dict = filter_common_receptors_multi_regulon(
        PDAC,
        selected_tfs=['SOX8', 'RXRA', 'SMAD2'],
        correlation_matrix=correlation_matrix,
        correlation_z=correlation_z,
        top_n_per_tf=10,
        commonality_threshold=2  # Must appear in at least 2
    )
    """
    
    from .plot_GRN import filter_receptors_by_correlation
    
    # Import the filtering function (you'll need to adjust this import based on your module structure)
    # For now, I'll assume it's available in the same module
    
    print(f"\n{'='*80}")
    print(f"Filtering common receptors across {len(selected_tfs)} regulons: {', '.join(selected_tfs)}")
    print(f"{'='*80}\n")
    
    # Step 1: Filter receptors for each TF independently
    tf_receptor_sets = {}
    tf_filtering_results = {}
    
    for tf in selected_tfs:
        print(f"\n--- Processing {tf} ---")
        
        # Get target genes for this TF
        regulon_dict = adata.uns.get('regulon_dict', {})
        tf_key = f"{tf}(+)"
        
        if tf_key not in regulon_dict:
            print(f"Warning: {tf_key} not found in regulon_dict, skipping...")
            continue
            
        target_genes = regulon_dict[tf_key]
        
        # Filter receptors for this TF
        try:
            results = filter_receptors_by_correlation(
                adata=adata,
                tf_name=tf,
                correlation_matrix=correlation_matrix,
                correlation_z=correlation_z,
                receptor_source=receptor_source,
                candidate_receptors=None,
                target_genes=target_genes,
                top_percent=top_percent_per_tf if top_n_per_tf is None else 1.0,
                min_correlation_threshold=min_correlation_threshold,
                min_z_threshold=min_z_threshold,
                top_n=top_n_per_tf,
                correlation_method=correlation_method,
                save_results=False
            )
            
            filtered_receptors = results['filtered_receptors']
            tf_receptor_sets[tf] = set(filtered_receptors)
            tf_filtering_results[tf] = results
            
            print(f"  → {len(filtered_receptors)} receptors passed filtering for {tf}")
            
        except Exception as e:
            print(f"Error filtering receptors for {tf}: {e}")
            continue
    
    if not tf_receptor_sets:
        raise ValueError("No TFs successfully filtered. Check your data and parameters.")
    
    # Step 2: Find common or union receptors
    print(f"\n{'='*80}")
    print(f"Computing receptor overlap...")
    print(f"{'='*80}\n")
    
    if return_union:
        # Union: combine all receptors from all TFs
        common_receptors = set()
        for receptors in tf_receptor_sets.values():
            common_receptors.update(receptors)
        print(f"Union of all receptors: {len(common_receptors)} total")
        
    else:
        # Count how many TFs each receptor appears in
        receptor_counts = {}
        for tf, receptors in tf_receptor_sets.items():
            for receptor in receptors:
                if receptor not in receptor_counts:
                    receptor_counts[receptor] = []
                receptor_counts[receptor].append(tf)
        
        # Set commonality threshold
        if commonality_threshold is None:
            commonality_threshold = len(selected_tfs)  # Must appear in ALL TFs
        
        # Filter receptors by commonality
        common_receptors = {
            receptor for receptor, tf_list in receptor_counts.items()
            if len(tf_list) >= commonality_threshold
        }
        
        print(f"Receptor commonality analysis:")
        print(f"  Commonality threshold: {commonality_threshold}/{len(selected_tfs)} TFs")
        print(f"  Total unique receptors across all TFs: {len(receptor_counts)}")
        print(f"  Receptors meeting threshold: {len(common_receptors)}")
        
        # Print detailed breakdown
        if len(receptor_counts) <= 50:  # Only print if manageable
            print(f"\nReceptor distribution:")
            for n_tfs in range(len(selected_tfs), 0, -1):
                receptors_at_n = [r for r, tfs in receptor_counts.items() if len(tfs) == n_tfs]
                if receptors_at_n:
                    print(f"  Present in {n_tfs}/{len(selected_tfs)} TFs: {len(receptors_at_n)} receptors")
                    if n_tfs >= commonality_threshold and len(receptors_at_n) <= 10:
                        print(f"    {', '.join(sorted(receptors_at_n))}")
    
    # Step 3: Create filtered receptor dictionary for each TF
    # Only include common receptors that were originally filtered for each TF
    filtered_receptor_dict = {}
    
    for tf in selected_tfs:
        if tf in tf_receptor_sets:
            # Intersection of common receptors with this TF's filtered receptors
            tf_common_receptors = list(common_receptors & tf_receptor_sets[tf])
            filtered_receptor_dict[tf] = tf_common_receptors
            print(f"\n{tf}: {len(tf_common_receptors)} common receptors")
            if len(tf_common_receptors) <= 15:
                print(f"  → {', '.join(sorted(tf_common_receptors))}")
    
    print(f"\n{'='*80}")
    print(f"Filtering complete!")
    print(f"{'='*80}\n")
    
    return filtered_receptor_dict


import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set


def plot_grn_network_with_common_receptors(
    adata,
    selected_tfs: List[str],
    correlation_matrix: pd.DataFrame,
    correlation_z: pd.DataFrame,
    min_importance: float = 0.1,
    receptor_source: str = 'receptor_dict',
    min_correlation_threshold: float = 0.05,
    min_z_threshold: float = 5.5,
    top_n_per_tf: Optional[int] = None,
    top_percent_per_tf: Optional[float] = 0.05,
    correlation_method: str = 'mean',
    commonality_threshold: int = None,
    **plot_kwargs
):
    """
    Plot GRN network with receptors common across multiple regulons.
    
    This is a fixed version that works around the receptor_source validation
    by temporarily replacing the existing receptor_dict instead of creating
    a new source.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    selected_tfs : list of str
        List of TF names to include
    correlation_matrix : pd.DataFrame
        Spatial correlation matrix
    correlation_z : pd.DataFrame  
        Spatial correlation z-scores
    min_importance : float
        Minimum edge importance for TF-target connections
    receptor_source : str
        Key in adata.uns for receptor data (default: 'receptor_dict')
    min_correlation_threshold : float
        Minimum correlation threshold for receptor filtering
    min_z_threshold : float
        Minimum z-score threshold for receptor filtering
    top_n_per_tf : int, optional
        Number of top receptors per TF to consider for commonality
    top_percent_per_tf : float, optional
        Percentage of top receptors per TF to consider
    correlation_method : str
        Aggregation method for correlations ('mean', 'median', 'max')
    commonality_threshold : int, optional
        Minimum number of TFs a receptor must appear in.
        If None, requires ALL TFs (strict intersection).
    **plot_kwargs
        Additional arguments passed to plot_grn_network
        
    Returns
    -------
    fig, G
        Figure and graph objects from plot_grn_network
        
    Example
    -------
    fig, G = plot_grn_network_with_common_receptors(
        PDAC,
        selected_tfs=['SOX9', 'RXRA', 'SMAD2'],
        correlation_matrix=correlation_matrix,
        correlation_z=correlation_z,
        min_importance=0.01,
        top_n_per_tf=10,
        commonality_threshold=None,
        figsize=(10, 8),
        show_heatmap=True
    )
    """
    
    from spagrn.plot_GRN import filter_common_receptors_multi_regulon, plot_grn_network
    
    # Step 1: Filter common receptors
    print("\n" + "="*80)
    print("FILTERING COMMON RECEPTORS")
    print("="*80)
    
    filtered_receptor_dict = filter_common_receptors_multi_regulon(
        adata=adata,
        selected_tfs=selected_tfs,
        correlation_matrix=correlation_matrix,
        correlation_z=correlation_z,
        receptor_source=receptor_source,
        min_correlation_threshold=min_correlation_threshold,
        min_z_threshold=min_z_threshold,
        top_n_per_tf=top_n_per_tf,
        top_percent_per_tf=top_percent_per_tf,
        correlation_method=correlation_method,
        commonality_threshold=commonality_threshold
    )
    
    # Check if we got any common receptors
    total_common = sum(len(receptors) for receptors in filtered_receptor_dict.values())
    if total_common == 0:
        print("\n" + "!"*80)
        print("WARNING: No common receptors found!")
        print("Try:")
        print("  1. Increasing top_n_per_tf (currently: {})".format(top_n_per_tf))
        print("  2. Lowering commonality_threshold (currently: {})".format(
            commonality_threshold if commonality_threshold else "ALL TFs"))
        print("  3. Lowering min_correlation_threshold (currently: {})".format(min_correlation_threshold))
        print("  4. Lowering min_z_threshold (currently: {})".format(min_z_threshold))
        print("!"*80 + "\n")
        return None, None
    
    # Step 2: Backup original receptor_dict
    original_receptor_dict = adata.uns.get(receptor_source, {}).copy()
    
    try:
        # Step 3: Temporarily replace with filtered receptors
        print("\n" + "="*80)
        print("PLOTTING NETWORK WITH COMMON RECEPTORS")
        print("="*80 + "\n")
        
        adata.uns[receptor_source] = filtered_receptor_dict
        
        # Step 4: Plot with standard receptor_source name
        fig, G = plot_grn_network(
            adata=adata,
            receptor_source=receptor_source,  # Use original name
            selected_tfs=selected_tfs,
            show_receptor=True,
            min_importance=min_importance,
            filter_receptor=False,  # Already filtered
            **plot_kwargs
        )
        
        print("\n" + "="*80)
        print("PLOTTING COMPLETE")
        print("="*80 + "\n")
        
        return fig, G
        
    finally:
        # Step 5: Restore original receptor_dict
        adata.uns[receptor_source] = original_receptor_dict
        print("Restored original receptor_dict")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Example 1: Basic usage with strict commonality (all TFs)
# ---------------------------------------------------------

from common_receptor_filtering_fixed import plot_grn_network_with_common_receptors

fig, G = plot_grn_network_with_common_receptors(
    PDAC,
    selected_tfs=['SOX9', 'RXRA', 'SMAD2'],
    correlation_matrix=correlation_matrix,
    correlation_z=correlation_z,
    min_importance=0.01,
    top_n_per_tf=10,  # Consider top 10 receptors per TF
    commonality_threshold=None,  # Must appear in ALL 3 TFs
    figsize=(10, 8),
    show_heatmap=True,
    show_legend=True,
    title='Common Receptors: SOX9, RXRA, SMAD2'
)


# Example 2: More permissive - receptors in at least 2 TFs
# ---------------------------------------------------------

fig, G = plot_grn_network_with_common_receptors(
    PDAC,
    selected_tfs=['SOX9', 'RXRA', 'SMAD2'],
    correlation_matrix=correlation_matrix,
    correlation_z=correlation_z,
    min_importance=0.01,
    top_n_per_tf=15,  # Larger pool
    commonality_threshold=2,  # At least 2 out of 3 TFs
    min_correlation_threshold=0.03,  # Lower threshold
    min_z_threshold=4.0,  # Lower threshold
    figsize=(10, 8),
    show_heatmap=True,
    title='Receptors in ≥2 Regulons'
)


# Example 3: Your exact use case
# -------------------------------

fig, G = plot_grn_network_with_common_receptors(
    PDAC,
    selected_tfs=['SOX9', 'RXRA', 'SMAD2'],
    correlation_matrix=correlation_matrix,
    correlation_z=correlation_z,
    min_importance=0.01,
    top_n_per_tf=10,
    top_percent_per_tf=1,
    min_correlation_threshold=0.05,
    min_z_threshold=5.5,
    commonality_threshold=None,
    correlation_method='mean',
    figsize=(10, 8),
    show_heatmap=True,
    show_score_dist=False
)


# Example 4: If you get no common receptors, try this
# ----------------------------------------------------

# Strategy: Start permissive, then tighten

# Step 1: Very permissive to see what's possible
fig, G = plot_grn_network_with_common_receptors(
    PDAC,
    selected_tfs=['SOX9', 'RXRA', 'SMAD2'],
    correlation_matrix=correlation_matrix,
    correlation_z=correlation_z,
    min_importance=0.01,
    top_n_per_tf=20,  # Large pool
    commonality_threshold=2,  # Just 2 out of 3
    min_correlation_threshold=0.03,  # Low threshold
    min_z_threshold=3.0,  # Low threshold
    figsize=(10, 8)
)

# Step 2: If that works, gradually tighten parameters
# Increase commonality_threshold to None (all 3 TFs)
# Decrease top_n_per_tf
# Increase min_correlation_threshold
# Increase min_z_threshold
"""



#################################################################################################################################################################################
"""
Spatial Correlation Analysis for Ligand-Receptor Genes vs Gene Groups

This module provides functions to:
1. Compute spatial correlations (using Moran's I framework) between ligand-receptor genes 
   and user-defined gene groups
2. Visualize results as bubble plots showing correlation strength and significance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import zscore
import warnings



def compute_lr_gene_group_correlations(
    adata,
    gene_groups: Dict[str, List[str]],
    lr_genes: Optional[Union[List[str], str]] = None,
    lr_csv_path: Optional[str] = None,
    correlation_matrix: Optional[pd.DataFrame] = None,
    correlation_z: Optional[pd.DataFrame] = None,
    layer_key: str = 'counts',
    latent_obsm_key: str = 'spatial',
    batch_key: Optional[str] = None,
    n_neighbors: int = 30,
    model: str = 'bernoulli',
    jobs: int = 4,
    aggregation_method: str = 'mean',
    compute_if_missing: bool = True,
    average_regulon: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute batch-aware spatial correlations between L/R genes and gene groups.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    gene_groups : dict
        Dictionary mapping group names to gene lists
    lr_genes : list or str, optional
        L/R genes to analyze. If a list, use as-is. If str like 'all_ligand', 'ligand', extract all ligands from lr_csv_path or adata.uns.
        Similarly for 'all_receptor', 'receptor'. If None, extracts both from lr_csv_path ('both' mode) or adata.uns.
    lr_csv_path : str, optional
        Path to interaction_LR.csv to extract L/R genes
    correlation_matrix : pd.DataFrame, optional
        Precomputed correlation matrix
    correlation_z : pd.DataFrame, optional
        Precomputed z-scores
    layer_key : str
        Layer containing count data
    latent_obsm_key : str
        Key for spatial coordinates
    batch_key : str
        REQUIRED if computing correlation matrix. Key for batch labels.
    n_neighbors : int
        Number of neighbors within each batch
    model : str
        Statistical model
    jobs : int
        Number of parallel jobs
    aggregation_method : str
        'mean', 'median', 'max', or 'min'
    compute_if_missing : bool
        Whether to compute correlation matrix if not provided
    average_regulon : bool, default True
        If True, aggregate correlations per group. If False, compute per individual gene in groups.
        
    Returns
    -------
    tuple
        (combined_df, corr_df, z_df, gene_type)
    """
    
    print("\n" + "="*80)
    print("BATCH-AWARE L/R GENE vs GENE GROUP CORRELATION ANALYSIS")
    print("="*80 + "\n")
    
    # Step 1: Get L/R genes
    gene_type = 'both'
    ligands = []
    receptors = []
    if isinstance(lr_genes, str):
        if lr_genes in ['all_ligand', 'ligand']:
            gene_type = 'ligand'
        elif lr_genes in ['all_receptor', 'receptor']:
            gene_type = 'receptor'
        else:
            raise ValueError("lr_genes str must be 'all_ligand', 'ligand', 'all_receptor', or 'receptor'")
        lr_genes = None  # Trigger extraction below
    
    if lr_genes is None:
        if lr_csv_path:
            print(f"Extracting {gene_type} from {lr_csv_path}...")
            result = extract_lr_genes_from_csv(lr_csv_path, gene_type=gene_type)
            # Handle different return types based on gene_type
            if gene_type == 'both':
                ligands, receptors, lr_genes = result
            else:
                # gene_type is 'ligand' or 'receptor', returns single list
                lr_genes = result
        else:
            print("No L/R genes provided. Attempting to extract from adata.uns...")
            lr_genes = _extract_lr_genes_from_adata(adata)
    
    # Filter to available genes
    available_lr_genes = [g for g in lr_genes if g in adata.var_names]
    print(f"\nL/R genes: {len(lr_genes)} provided, {len(available_lr_genes)} found in adata")
    
    if len(available_lr_genes) == 0:
        raise ValueError("No L/R genes found in adata")
    
    # Step 2: Validate and filter gene groups
    filtered_gene_groups = {}
    for group_name, genes in gene_groups.items():
        available_genes = [g for g in genes if g in adata.var_names]
        if len(available_genes) > 0:
            filtered_gene_groups[group_name] = available_genes
            print(f"  {group_name}: {len(available_genes)}/{len(genes)} genes available")
    
    if len(filtered_gene_groups) == 0:
        raise ValueError("No gene groups have genes in adata")
    
    # Step 3: Get or compute batch-aware correlation matrix
    if correlation_matrix is None or correlation_z is None:
        if not compute_if_missing:
            raise ValueError(
                "correlation_matrix and correlation_z required, or set compute_if_missing=True"
            )
        
        if batch_key is None:
            raise ValueError(
                "batch_key is REQUIRED for computing batch-aware correlations. "
                "Provide the column name in adata.obs with batch/patient/sample labels."
            )
        
        print("\nComputing batch-aware spatial correlation matrix (focused subset)...")
        # Only the genes actually indexed below are needed: the L/R rows plus every
        # group gene used as a column. This avoids the all-genes O(n^2) computation.
        involved = list(dict.fromkeys(
            list(available_lr_genes)
            + [g for genes in filtered_gene_groups.values() for g in genes]
        ))
        correlation_matrix, correlation_z = compute_global_spatial_correlation_matrix(
            adata=adata,
            layer_key=layer_key,
            latent_obsm_key=latent_obsm_key,
            batch_key=batch_key,
            n_neighbors=n_neighbors,
            model=model,
            jobs=jobs,
            genes=involved,
        )
    else:
        print("\nUsing provided correlation matrices")
        if batch_key:
            print(f"  Note: batch_key='{batch_key}' provided")
            print(f"  Assuming correlation matrices were computed with batch awareness")
    
    # Step 4: Compute aggregated correlations
    print(f"\nComputing {aggregation_method} correlations...")
    print(f"  L/R genes: {len(available_lr_genes)}")
    print(f"  Gene groups: {len(filtered_gene_groups)}")
    
    results_corr = []
    results_z = []
    
    for lr_gene in available_lr_genes:
        if lr_gene not in correlation_matrix.index:
            continue
        
        row_corr = {}
        row_z = {}
        
        for group_name, group_genes in filtered_gene_groups.items():
            if average_regulon:
                corr_values = []
                z_values = []
                for gene in group_genes:
                    if gene in correlation_matrix.columns:
                        corr_values.append(correlation_matrix.loc[lr_gene, gene])
                        z_values.append(correlation_z.loc[lr_gene, gene])
                if corr_values:
                    if aggregation_method == 'mean':
                        row_corr[group_name] = np.mean(corr_values)
                        row_z[group_name] = np.mean(z_values)
                    elif aggregation_method == 'median':
                        row_corr[group_name] = np.median(corr_values)
                        row_z[group_name] = np.median(z_values)
                    elif aggregation_method == 'max':
                        max_idx = np.argmax(np.abs(corr_values))
                        row_corr[group_name] = corr_values[max_idx]
                        row_z[group_name] = z_values[max_idx]
                    elif aggregation_method == 'min':
                        min_idx = np.argmin(np.abs(corr_values))
                        row_corr[group_name] = corr_values[min_idx]
                        row_z[group_name] = z_values[min_idx]
            else:
                for gene in group_genes:
                    if gene in correlation_matrix.columns:
                        row_corr[f"{group_name}_{gene}"] = correlation_matrix.loc[lr_gene, gene]
                        row_z[f"{group_name}_{gene}"] = correlation_z.loc[lr_gene, gene]
        
        results_corr.append(row_corr)
        results_z.append(row_z)
    
    # Create DataFrames
    corr_df = pd.DataFrame(results_corr, index=available_lr_genes)
    z_df = pd.DataFrame(results_z, index=available_lr_genes)
    
    # Add type column
    gene_type_map = {}
    for g in ligands:
        gene_type_map[g] = 'Ligand'
    for g in receptors:
        gene_type_map[g] = 'Receptor'
    corr_df['type'] = [gene_type_map.get(idx, 'Unknown') for idx in corr_df.index]
    z_df['type'] = [gene_type_map.get(idx, 'Unknown') for idx in z_df.index]
    
    # Combined DataFrame
    combined_df = corr_df.copy()
    for col in z_df.columns:
        if col != 'type':
            combined_df[f"{col}_zscore"] = z_df[col]
    combined_df['type'] = corr_df['type']
    
    print("\n" + "="*80)
    print("COMPUTATION COMPLETE")
    print("="*80)
    print(f"Result shape: {corr_df.shape}")
    
    return combined_df, corr_df, z_df, gene_type




def plot_lr_gene_group_bubble(
    corr_df: pd.DataFrame,
    z_df: pd.DataFrame,
    top_n: int = 20,
    selection_method: str = 'positive',
    average_regulon: bool = True,
    plot_raw: bool = True,
    group_by_lr: bool = False,
    max_dot_size: int = 200,
    figsize: Tuple[int, int] = (12, 10),
    cmap_pos: str = 'Reds',
    cmap_neg: str = 'Blues_r',
    size_scale: float = 500,
    alpha: float = 0.8,
    title: Optional[str] = 'L/R Gene-Group Correlations',
    xlabel: str = 'Gene Groups',
    ylabel: str = 'L/R Genes',
    xlabel_size: int = 12,
    ylabel_size: int = 12,
    xtick_size: int = 10,
    ytick_size: int = 10,
    show_colorbar: bool = True,
    show_grid: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    z_threshold: float = 2.0,
    corr_threshold: float = 0.01,
    annotate_values: bool = False,
    font_scale: float = 1.0,
    gene_groups: Optional[Dict[str, List[str]]] = None,  # Optional to pass for x grouping when not average
    xtick_rotation: int = 45  # Rotation angle for x-axis tick labels
) -> plt.Figure:
    """
    Create a bubble plot for L/R gene-group correlations.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation DataFrame (rows: L/R genes, cols: groups)
    z_df : pd.DataFrame
        Z-score DataFrame
    top_n : int
        Number of top genes to display
    selection_method : str
        'positive', 'negative', or 'abs_mean'
    average_regulon : bool
        If False, assumes corr_df has per-gene columns (e.g., 'group_gene')
    plot_raw : bool
        Use raw correlations (True) or z-scores (False)
    group_by_lr : bool
        Group y-axis by 'type' column if present
    max_dot_size : int
        Max bubble size
    figsize : tuple
        Figure size
    # ... other params with defaults ...
    gene_groups : dict, optional
        Original gene_groups to help with x-grouping when average_regulon=False

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    # Set seaborn style
    sns.set(font_scale=font_scale)

    # Select data source for plotting
    source_df = corr_df if plot_raw else z_df
    threshold = corr_threshold if plot_raw else z_threshold

    # Drop 'type' if present for both raw and plotting dataframes
    corr_df_clean = corr_df.drop(columns=['type']) if 'type' in corr_df.columns else corr_df
    plot_df = source_df.drop(columns=['type']) if 'type' in source_df.columns else source_df

    # Gene selection - ALWAYS based on RAW correlation scores (corr_df), regardless of plot_raw
    # and always based on averaged gene group scores
    if not average_regulon:
        # When average_regulon=False, columns are like 'SMAD2(+)_SMAD2', 'SMAD2(+)_GSTP1', etc.
        # We need to average across each gene group for selection
        if gene_groups is not None:
            # Create temporary averaged dataframe for selection using RAW correlations
            temp_avg_df = pd.DataFrame()
            for group_name in gene_groups.keys():
                group_cols = [col for col in corr_df_clean.columns if col.startswith(group_name + '_')]
                if len(group_cols) > 0:
                    temp_avg_df[group_name] = corr_df_clean[group_cols].mean(axis=1)
            gene_scores = temp_avg_df.mean(axis=1)
        else:
            # Fallback to regular mean if no gene_groups provided
            gene_scores = corr_df_clean.mean(axis=1)
    else:
        # When average_regulon=True, columns are already gene groups, use RAW correlations
        gene_scores = corr_df_clean.mean(axis=1)
    
    # Apply selection method on RAW correlation scores
    if selection_method == 'positive':
        selected_genes = gene_scores.nlargest(top_n).index.tolist()
    elif selection_method == 'negative':
        selected_genes = gene_scores.nsmallest(top_n).index.tolist()
    elif selection_method == 'abs_mean':
        if not average_regulon and gene_groups is not None:
            temp_avg_df = pd.DataFrame()
            for group_name in gene_groups.keys():
                group_cols = [col for col in corr_df_clean.columns if col.startswith(group_name + '_')]
                if len(group_cols) > 0:
                    temp_avg_df[group_name] = corr_df_clean[group_cols].mean(axis=1)
            gene_scores = temp_avg_df.abs().mean(axis=1)
        else:
            gene_scores = corr_df_clean.abs().mean(axis=1)
        selected_genes = gene_scores.nlargest(top_n).index.tolist()
    else:
        raise ValueError("Invalid selection_method")

    # Reverse gene order so highest values appear at TOP of y-axis
    selected_genes = selected_genes[::-1]
    
    plot_df = plot_df.loc[selected_genes]

    # Group by LR for y-axis
    if group_by_lr and 'type' in source_df.columns:
        types = source_df['type'].loc[selected_genes]
        plot_df['type'] = types.reset_index(drop=True)
        plot_df = plot_df.sort_values('type')
        sorted_genes = plot_df.index
        plot_df = plot_df.drop(columns=['type'])
        # Count ligands
        lig_count = (types == 'Ligand').sum()

    # Melt for plotting
    melted = plot_df.reset_index().melt(id_vars='index', var_name='group', value_name='value')
    # Calculate raw sizes - DON'T clip here, let seaborn's sizes parameter handle the range
    melted['size'] = np.abs(melted['value']) * size_scale

    # For x grouping if not average_regulon
    if not average_regulon:
        melted['regulon'] = melted['group'].apply(lambda x: x.split('_')[0] if '_' in x else x)
        melted = melted.sort_values(['regulon', 'group'])
        # Unique order for x
        group_order = melted['group'].unique()
        # To add lines and labels
        regulons = sorted(set(melted['regulon']))
        regulon_centers = {}
        cum = 0
        for reg in regulons:
            group_len = sum(1 for col in plot_df.columns if col.startswith(reg + '_'))
            center = cum + group_len / 2 - 0.5
            regulon_centers[reg] = center
            cum += group_len

    # For colorbar range, use all values
    all_values = melted['value'].dropna()
    if all_values.empty:
        min_v, max_v = -1, 1
    else:
        min_v, max_v = all_values.min(), all_values.max()

    # Filter by threshold
    filtered_melted = melted[melted['value'].abs() > threshold].copy()

    # Create figure with white background
    # Create figure with constrained_layout for better automatic spacing
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', constrained_layout=False)
    
    # Calculate dynamic positioning based on figure size
    fig_width, fig_height = figsize
    
    # Dynamic colorbar padding: slightly larger base padding
    cbar_pad = max(0.05, min(0.20, 0.20 * (fig_width / 12)))
    
    ax.set_facecolor('white')

    if filtered_melted.empty:
        ax.text(0.5, 0.5, "No values above threshold", ha='center', va='center', transform=ax.transAxes)
    else:
        # Determine value range for symmetric colormap
        abs_max = max(abs(min_v), abs(max_v))
        vmin, vmax = -abs_max, abs_max
        
        sns.scatterplot(
            data=filtered_melted,
            x='group',
            y='index',
            size='size',
            hue='value',
            palette='RdBu_r',
            alpha=alpha,
            ax=ax,
            sizes=(10, max_dot_size),
            hue_norm=(vmin, vmax),
            legend=False  # We'll create custom legend
        )

    # Set x ticks order if not average
    _ha = 'center' if xtick_rotation == 90 else 'right'
    if not average_regulon:
        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(group_order, rotation=xtick_rotation, ha=_ha)
    else:
        # For average regulon, align based on rotation
        plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha=_ha)

    # Add x group lines and labels if not average_regulon
    if not average_regulon:
        cum = 0
        for i, reg in enumerate(regulons):
            # Skip the first vertical line (before first group)
            if i > 0:
                ax.axvline(cum - 0.5, color='black', ls='-', lw=1.5)
            center = regulon_centers[reg]
            # Move gene group labels closer to plot (changed from 1.05 to 1.02)
            ax.text(center, 1.02, reg, ha='center', va='bottom', transform=ax.get_xaxis_transform())
            group_len = sum(1 for col in plot_df.columns if col.startswith(reg + '_'))
            cum += group_len

    # Add y group lines if group_by_lr
    if group_by_lr and 'type' in source_df.columns:
        ax.axhline(lig_count - 0.5, color='black', ls='-', lw=1.5)
        total = len(selected_genes)
        ax.text(-0.05, lig_count / 2 - 0.5, 'Ligand', rotation=90, va='center', ha='right', transform=ax.get_yaxis_transform())
        ax.text(-0.05, lig_count + (total - lig_count) / 2 - 0.5, 'Receptor', rotation=90, va='center', ha='right', transform=ax.get_yaxis_transform())

    # Annotations
    if annotate_values and not filtered_melted.empty:
        for _, row in filtered_melted.iterrows():
            ax.text(row['group'], row['index'], f"{row['value']:.2f}", ha='center', va='center', fontsize=8)

    # Move title to the very top with more space above the plot
    ax.set_title(title, fontsize=xlabel_size + 2, pad=25)  # Increased pad to move title higher
    ax.set_xlabel(xlabel, fontsize=xlabel_size)
    ax.set_ylabel(ylabel, fontsize=ylabel_size)
    
    # Reduce y-axis padding
    ax.set_ylim(-0.8, len(plot_df.index) - 0.2)

    if show_colorbar and not filtered_melted.empty:
        # Create symmetric colorbar centered at 0
        abs_max = max(abs(min_v), abs(max_v))
        norm = plt.Normalize(-abs_max, abs_max)
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
        sm.set_array([])
        
        # Position colorbar to the right of the plot with dynamic padding
        cbar = fig.colorbar(sm, ax=ax, label='Correlation' if plot_raw else 'Z-score', 
                           pad=cbar_pad, aspect=30, shrink=0.8)
        
        # Remove any existing legends that seaborn might have created
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        # Add size legend with 3 representative sizes based on absolute VALUES
        # The legend should reflect the actual values being plotted (raw or z-scores)
        
        # Use filtered_melted for the VALUES to show in legend
        abs_values = filtered_melted['value'].abs()
        
        if not abs_values.empty:
            # Get the range of absolute values in the DISPLAYED data
            abs_min, abs_max = abs_values.min(), abs_values.max()
            abs_mid = (abs_min + abs_max) / 2
            
            legend_abs_values = [abs_min, abs_mid, abs_max]
            
            # CRITICAL: Seaborn normalizes the 'size' column values (NOT clipped) to fit sizes=(10, max_dot_size)
            # The size column contains: abs(value) * size_scale (unclipped)
            # Seaborn maps [min(size), max(size)] → [10, max_dot_size]
            
            # Get the actual size range from the FILTERED data (what seaborn sees)
            size_min = filtered_melted['size'].min()
            size_max = filtered_melted['size'].max()
            
            if abs_max > 0 and size_max > size_min:
                # Calculate the raw sizes for our legend values (same formula as main plot)
                legend_raw_sizes = [
                    abs_min * size_scale,
                    abs_mid * size_scale,
                    abs_max * size_scale
                ]
                
                # Now normalize them the way seaborn does: map [size_min, size_max] → [10, max_dot_size]
                # Formula: normalized = 10 + (size - size_min) / (size_max - size_min) * (max_dot_size - 10)
                legend_sizes = []
                for s in legend_raw_sizes:
                    normalized = 10 + (s - size_min) / (size_max - size_min) * (max_dot_size - 10)
                    legend_sizes.append(normalized)
                
                # Ensure minimum size is at least 10
                legend_sizes = [max(s, 10) for s in legend_sizes]
            else:
                legend_sizes = [10, max_dot_size/2, max_dot_size]
            
            # Create size legend elements
            from matplotlib.lines import Line2D
            legend_elements = []
            for i, (size, abs_val) in enumerate(zip(legend_sizes, legend_abs_values)):
                label = f'{abs_val:.2f}' if plot_raw else f'{abs_val:.1f}'
                # Properly convert scatter size to markersize
                # seaborn/matplotlib scatter uses size in points^2, legend marker uses diameter in points
                marker_size = np.sqrt(size)
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='gray', markersize=marker_size,
                          label=label, alpha=alpha, linestyle='None',
                          markeredgewidth=0)
                )
            
            # Add size legend to the right, vertically aligned with the colorbar
            legend_title = 'Absolute Correlation' if plot_raw else 'Absolute Scaled Correlation'
            
            # Position legend to the right of the plot, vertically centered
            # bbox_to_anchor: (x, y) where x > 1 places it outside the plot on the right
            # loc='center left' anchors the legend's left side to the bbox point
            ax.legend(handles=legend_elements, 
                     title=legend_title,
                     loc='center left',  # Changed from 'upper center'
                     bbox_to_anchor=(1.15, -0.2),  # Position on right, vertically centered with colorbar
                     ncol=1,  # Changed from 3 to 1 for vertical stacking
                     frameon=True,
                     fontsize=9,
                     title_fontsize=10,
                     facecolor='white')

    if show_grid:
        ax.grid(True, alpha=0.3, color='lightgrey', linestyle='-', linewidth=0.5)
    
    # Add black frame around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    plt.xticks(rotation=xtick_rotation, ha=_ha, fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    
    # Use subplots_adjust with larger margins to prevent overlapping
    # bottom: space for rotated x-labels + legend
    # right: space for y-labels + colorbar  
    # left: space for y-labels
    # top: space for title (reduced to 0.92 to accommodate higher title position)
    plt.subplots_adjust(left=0.15, bottom=0.28, right=0.77 if show_colorbar else 0.95, top=0.92)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    print(f"Plotting {len(selected_genes)} genes across {len(plot_df.columns)} groups")

    return fig



def _extract_lr_genes_from_adata(adata) -> List[str]:
    """Extract ligand-receptor genes from adata.uns."""
    lr_genes = set()
    
    # Try receptor_dict
    if 'receptor_dict' in adata.uns:
        receptor_dict = adata.uns['receptor_dict']
        for tf, receptors in receptor_dict.items():
            lr_genes.update(receptors)
        print(f"  Found {len(lr_genes)} receptors from receptor_dict")
    
    # Try ligand_list
    if 'ligand_list' in adata.uns:
        ligands = adata.uns['ligand_list']
        if isinstance(ligands, list):
            lr_genes.update(ligands)
        elif isinstance(ligands, dict):
            for ligand_set in ligands.values():
                lr_genes.update(ligand_set)
        print(f"  Total {len(lr_genes)} LR genes after adding ligands")
    
    # Try receptor_dict_all
    if 'receptor_dict_all' in adata.uns:
        receptor_dict_all = adata.uns['receptor_dict_all']
        for tf, receptors in receptor_dict_all.items():
            lr_genes.update(receptors)
        print(f"  Total {len(lr_genes)} LR genes after adding receptor_dict_all")
    
    return sorted(list(lr_genes))


# ============================================================================
# CONVENIENCE WRAPPER
# ============================================================================

def analyze_lr_gene_groups(
    adata,
    gene_groups: Dict[str, List[str]],
    lr_genes: Optional[List[str]] = None,
    correlation_matrix: Optional[pd.DataFrame] = None,
    correlation_z: Optional[pd.DataFrame] = None,
    top_n: int = 20,
    selection_method: str = 'abs_mean',
    figsize: Tuple[int, int] = (12, 10),
    z_threshold: Optional[float] = 2.0,
    corr_threshold: Optional[float] = None,
    save_prefix: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, plt.Figure]:
    """
    One-shot function to compute correlations and create bubble plot.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    gene_groups : dict
        Gene groups to analyze
    lr_genes : list, optional
        LR genes to analyze (auto-extracted if None)
    correlation_matrix : pd.DataFrame, optional
        Precomputed correlation matrix
    correlation_z : pd.DataFrame, optional
        Precomputed z-scores
    top_n : int
        Number of top genes to plot
    selection_method : str
        Gene selection method
    figsize : tuple
        Figure size
    z_threshold : float, optional
        Z-score threshold for plotting
    corr_threshold : float, optional
        Correlation threshold for plotting
    save_prefix : str, optional
        Prefix for saving results
    **kwargs
        Additional arguments for compute or plot functions
        
    Returns
    -------
    tuple
        (combined_df, corr_df, z_df, fig)
        
    Example
    -------
    gene_groups = {
        'SMAD2_module': ['SMAD2', 'NEAT1', 'IFITM3'],
        'SOX9_module': ['KRT17', 'KRT7', 'SOX9']
    }
    
    combined_df, corr_df, z_df, fig = analyze_lr_gene_groups(
        adata=PDAC,
        gene_groups=gene_groups,
        correlation_matrix=correlation_matrix,
        correlation_z=correlation_z,
        top_n=20,
        z_threshold=2.0,
        save_prefix='lr_analysis'
    )
    """
    
    # Compute correlations
    combined_df, corr_df, z_df, gene_type = compute_lr_gene_group_correlations(
        adata=adata,
        gene_groups=gene_groups,
        lr_genes=lr_genes,
        correlation_matrix=correlation_matrix,
        correlation_z=correlation_z,
        **{k: v for k, v in kwargs.items() if k in [
            'layer_key', 'latent_obsm_key', 'batch_key', 'n_neighbors',
            'model', 'jobs', 'aggregation_method', 'compute_if_missing'
        ]}
    )
    
    # Determine default labels based on gene_type if not provided in kwargs
    if 'title' not in kwargs:
        if gene_type == 'ligand':
            kwargs['title'] = f'Top {top_n} Ligand-Gene Group Correlations'
        elif gene_type == 'receptor':
            kwargs['title'] = f'Top {top_n} Receptor-Gene Group Correlations'
        else:
            kwargs['title'] = f'Top {top_n} L/R Gene-Group Correlations'
    
    if 'ylabel' not in kwargs:
        if gene_type == 'ligand':
            kwargs['ylabel'] = 'Ligand'
        elif gene_type == 'receptor':
            kwargs['ylabel'] = 'Receptor'
        else:
            kwargs['ylabel'] = 'Ligand/Receptor'
    
    # Create plot
    fig = plot_lr_gene_group_bubble(
        corr_df=corr_df,
        z_df=z_df,
        top_n=top_n,
        selection_method=selection_method,
        figsize=figsize,
        z_threshold=z_threshold,
        corr_threshold=corr_threshold,
        save_path=f"{save_prefix}_bubble.pdf" if save_prefix else None,
        **{k: v for k, v in kwargs.items() if k in [
            'cmap_pos', 'cmap_neg', 'size_scale', 'alpha', 'title',
            'xlabel', 'ylabel', 'show_colorbar', 'show_grid', 'dpi',
            'annotate_values', 'font_scale'
        ]}
    )
    
    # Save data if requested
    if save_prefix:
        combined_df.to_csv(f"{save_prefix}_combined.csv")
        corr_df.to_csv(f"{save_prefix}_correlations.csv")
        z_df.to_csv(f"{save_prefix}_zscores.csv")
        print(f"\nSaved data tables to {save_prefix}_*.csv")
    
    return combined_df, corr_df, z_df, fig



"""
Batch-Aware Spatial Correlation Analysis for Ligand-Receptor Genes vs Gene Groups

This module provides functions to:
1. Extract ligand and receptor genes from CellChat interaction database
2. Compute batch-aware spatial correlations (cells from different batches are never neighbors)
3. Visualize results as bubble plots showing correlation strength and significance

CRITICAL: Spatial neighbors are computed within-batch only - cells from different 
batches/patients/samples are NEVER considered neighbors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import zscore
import warnings


def extract_lr_genes_from_csv(
    lr_csv_path: str,
    gene_type: str = 'both'
) -> Union[List[str], Tuple[List[str], List[str]]]:
    """
    Extract ligand and/or receptor genes from CellChat interaction database.
    
    Parameters
    ----------
    lr_csv_path : str
        Path to interaction_LR.csv file
    gene_type : str
        Which genes to extract: 'ligand', 'receptor', or 'both'
        
    Returns
    -------
    list or tuple
        If gene_type='both': (ligands_list, receptors_list, all_lr_genes)
        Otherwise: single list of genes
        
    Example
    -------
    ligands, receptors, all_lr = extract_lr_genes_from_csv(
        'interaction_LR.csv', 
        gene_type='both'
    )
    """
    
    df = pd.read_csv(lr_csv_path)
    
    # Extract ligands (single genes)
    ligands = set(df['ligand'].dropna().unique())
    
    # Extract receptors (may be complex like "TGFBR1_TGFBR2")
    receptor_complexes = df['receptor'].dropna().unique()
    receptors = set()
    
    for complex_str in receptor_complexes:
        # Split by common separators
        genes = complex_str.replace('_', ' ').replace('+', ' ').replace(',', ' ').split()
        receptors.update(genes)
    
    print(f"Extracted {len(ligands)} unique ligands")
    print(f"Extracted {len(receptors)} unique receptors")
    
    if gene_type == 'ligand':
        return sorted(list(ligands))
    elif gene_type == 'receptor':
        return sorted(list(receptors))
    elif gene_type == 'both':
        all_lr = sorted(list(ligands | receptors))
        print(f"Total unique L/R genes: {len(all_lr)}")
        return sorted(list(ligands)), sorted(list(receptors)), all_lr
    else:
        raise ValueError(f"gene_type must be 'ligand', 'receptor', or 'both', got {gene_type}")
    
    
    






def analyze_lr_gene_groups_batch_aware(
    adata,
    gene_groups: Dict[str, List[str]],
    lr_csv_path: Optional[str] = None,
    lr_genes: Optional[Union[List[str], str]] = None,
    correlation_matrix: Optional[pd.DataFrame] = None,
    correlation_z: Optional[pd.DataFrame] = None,
    batch_key: Optional[str] = None,
    top_n: int = 20,
    selection_method: str = 'positive',
    average_regulon: bool = True,
    plot_raw: bool = True,
    group_by_lr: bool = False,
    max_dot_size: int = 200,
    figsize: Tuple[int, int] = (12, 10),
    z_threshold: Optional[float] = 2.0,
    save_prefix: Optional[str] = None,
    xlabel_size: int = 12,
    ylabel_size: int = 12,
    xtick_size: int = 10,
    ytick_size: int = 10,
    xtick_rotation: int = 45,  # Rotation angle for x-axis tick labels
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, plt.Figure]:
    """
    One-shot batch-aware L/R gene-group correlation analysis.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    gene_groups : dict
        Gene groups to analyze
    lr_csv_path : str, optional
        Path to interaction_LR.csv for extracting L/R genes
    lr_genes : list or str, optional
        Explicit L/R genes (overrides lr_csv_path). If str like 'all_ligand' or 'all_receptor', extract accordingly.
    correlation_matrix : pd.DataFrame, optional
        Precomputed batch-aware correlation matrix
    correlation_z : pd.DataFrame, optional
        Precomputed z-scores
    batch_key : str
        Column in adata.obs with batch labels (REQUIRED if computing correlations)
    top_n : int
        Number of top genes to plot
    selection_method : str, default 'positive'
        'positive': Top N by positive avg corr. 'negative': Top N by negative avg. 'abs_mean': Top by abs avg.
    average_regulon : bool, default True
        If True, average per regulon (gene group). If False, plot each gene in gene_groups, grouped by regulon.
    plot_raw : bool, default True
        If True, plot raw correlations. If False, plot z-scores.
    group_by_lr : bool, default False
        If True, group y-axis genes by 'Ligand' and 'Receptor' (requires 'type' column in corr_df).
    max_dot_size : int, default 200
        Maximum dot size to prevent oversized bubbles.
    figsize : tuple
        Figure size
    z_threshold : float
        Z-score threshold for plotting
    save_prefix : str, optional
        Prefix for saving results
    xlabel_size : int, default 12
        Font size for x-axis label
    ylabel_size : int, default 12
        Font size for y-axis label
    xtick_size : int, default 10
        Font size for x-axis tick labels
    ytick_size : int, default 10
        Font size for y-axis tick labels
    **kwargs
        Additional arguments for compute_lr_gene_group_correlations or plot_lr_gene_group_bubble.
        
    Returns
    -------
    tuple
        (combined_df, corr_df, z_df, fig)
        
    Example
    -------
    # With batch key specified
    gene_groups = {
        'SMAD2_module': ['SMAD2', 'NEAT1', 'IFITM3'],
        'SOX9_module': ['KRT17', 'KRT7', 'SOX9']
    }
    
    combined, corr, z, fig = analyze_lr_gene_groups_batch_aware(
        adata=PDAC,
        gene_groups=gene_groups,
        lr_csv_path='interaction_LR.csv',
        batch_key='patient',  # CRITICAL for batch awareness
        correlation_matrix=corr_matrix,  # Pre-computed with batch awareness
        correlation_z=corr_z,
        top_n=20,
        save_prefix='batch_aware_lr'
    )
    """
    # Compute correlations
    combined_df, corr_df, z_df, gene_type = compute_lr_gene_group_correlations(
        adata=adata,
        gene_groups=gene_groups,
        lr_genes=lr_genes,
        lr_csv_path=lr_csv_path,
        correlation_matrix=correlation_matrix,
        correlation_z=correlation_z,
        layer_key=kwargs.get('layer_key', 'counts'),
        latent_obsm_key=kwargs.get('latent_obsm_key', 'spatial'),
        batch_key=batch_key,
        n_neighbors=kwargs.get('n_neighbors', 30),
        model=kwargs.get('model', 'bernoulli'),
        jobs=kwargs.get('jobs', 4),
        aggregation_method=kwargs.get('aggregation_method', 'mean'),
        compute_if_missing=kwargs.get('compute_if_missing', True),
        average_regulon=average_regulon
    )
    
    # Determine labels based on gene_type
    if gene_type == 'ligand':
        default_title = f'Top {top_n} Ligand-Gene Group Correlations'
        default_ylabel = 'Ligand'
    elif gene_type == 'receptor':
        default_title = f'Top {top_n} Receptor-Gene Group Correlations'
        default_ylabel = 'Receptor'
    else:  # 'both' or other cases
        default_title = f'Top {top_n} L/R Gene-Group Correlations'
        default_ylabel = 'Ligand/Receptor'
    
    # Create plot
    fig = plot_lr_gene_group_bubble(
        corr_df=corr_df,
        z_df=z_df,
        top_n=top_n,
        selection_method=selection_method,
        average_regulon=average_regulon,
        plot_raw=plot_raw,
        group_by_lr=group_by_lr,
        max_dot_size=max_dot_size,
        figsize=figsize,
        z_threshold=z_threshold,
        corr_threshold=kwargs.get('corr_threshold', 0.01),
        save_path=f"{save_prefix}_bubble.pdf" if save_prefix else None,
        cmap_pos=kwargs.get('cmap_pos', 'Reds'),
        cmap_neg=kwargs.get('cmap_neg', 'Blues_r'),
        size_scale=kwargs.get('size_scale', 500),
        alpha=kwargs.get('alpha', 0.8),
        title=kwargs.get('title', default_title),
        xlabel=kwargs.get('xlabel', 'Gene Groups'),
        ylabel=kwargs.get('ylabel', default_ylabel),
        xlabel_size=xlabel_size,
        ylabel_size=ylabel_size,
        xtick_size=xtick_size,
        ytick_size=ytick_size,
        xtick_rotation=xtick_rotation,
        show_colorbar=kwargs.get('show_colorbar', True),
        show_grid=kwargs.get('show_grid', True),
        dpi=kwargs.get('dpi', 300),
        annotate_values=kwargs.get('annotate_values', False),
        font_scale=kwargs.get('font_scale', 1.0),
        gene_groups=gene_groups
    )
    
    # Save data
    if save_prefix:
        combined_df.to_csv(f"{save_prefix}_combined.csv")
        corr_df.to_csv(f"{save_prefix}_correlations.csv")
        z_df.to_csv(f"{save_prefix}_zscores.csv")
        print(f"\nSaved results to {save_prefix}_*.csv")
    
    return combined_df, corr_df, z_df, fig

# ==========================================================================
# Receptor -> regulon spatial correlation (potential upstream receptors)
# ==========================================================================
def compute_receptor_regulon_correlations(
    adata,
    niche_df,
    layer_key: str = 'counts',
    latent_obsm_key: str = 'spatial',
    batch_key: Optional[str] = None,
    n_neighbors: int = 100,
    model: str = 'bernoulli',
    receptor_key: str = 'to',
    jobs: int = 4,
    hs=None,
    adjacencies=None,
    regulon_dict=None,
    save_to_uns: bool = True,
    save_path: Optional[str] = None,
    cache: bool = True,
):
    """
    Spatial correlation between each regulon's POTENTIAL upstream receptors and
    that regulon's TF + pruned target genes.

    "Potential upstream receptors" = genes that are (a) in the ligand-receptor
    database receptor list, and (b) linked to the TF in the PRE-pruning
    adjacencies (adata.uns['adj'], i.e. before cisTarget motif lookup). These are
    candidate upstream receptors that motif pruning would normally discard because
    a receptor rarely carries the TF's motif.

    For each regulon, the package's batch-corrected pairwise spatial statistic
    (compute_hs_pairs_centered_cond — the same one filter_receptors_by_correlation
    was built around, so min_correlation_threshold / min_z_threshold keep their
    meaning) is computed on the small gene set {TF} + {its potential receptors} +
    {its pruned targets}. The receptor x {TF, targets} cells are kept.

    Writes (and returns):
      adata.uns['receptor_corr']           - correlation matrix  (receptors x cols)
      adata.uns['receptor_corr_z']         - z-scores            (receptors x cols)
      adata.uns['potential_receptor_dict'] - {tf: [potential receptors]}
    where cols = the TFs and their pruned targets.

    Parameters mirror compute_global_spatial_correlation_matrix; pass `hs` to reuse
    an already-built Hotspot graph (infer() does this) instead of rebuilding it.
    """
    import itertools

    # ---- cache ----
    if cache and save_path:
        cpath, zpath = f"{save_path}_receptor_corr.csv", f"{save_path}_receptor_corr_z.csv"
        if os.path.exists(cpath) and os.path.exists(zpath):
            print(f"Loading cached receptor-regulon correlations from {save_path}")
            corr = pd.read_csv(cpath, index_col=0)
            zsc = pd.read_csv(zpath, index_col=0)
            if save_to_uns:
                adata.uns['receptor_corr'] = corr
                adata.uns['receptor_corr_z'] = zsc
            return corr, zsc

    regulon_dict = adata.uns['regulon_dict'] if regulon_dict is None else regulon_dict

    # ---- batch correction is mandatory when multiple samples exist ----
    # (the spatial graph must never connect cells from different samples)
    if batch_key is None and (hs is None):
        n_samp = None
        for k in ('patient', 'sample', 'batch'):
            if k in adata.obs:
                n_samp = adata.obs[k].nunique()
                break
        if n_samp is None or n_samp > 1:
            raise ValueError(
                "batch_key is required: with multiple samples the spatial graph must be "
                "built within each sample (no cross-sample neighbours). Pass batch_key="
                "'patient' (or your sample column), or pass a pre-built batch-aware hs."
            )

    var = set(adata.var_names)
    receptor_pool = set(map(str, niche_df[receptor_key])) if niche_df is not None else set()

    # ---- candidate receptors = ALL L/R-database receptors present in the data ----
    # (receptors are excluded by the upfront spatial-gene selection, so they are NOT
    #  in adj/modules; the only complete candidate pool is the LR receptors in var_names)
    receptors_in_data = sorted(receptor_pool & var)
    if not receptors_in_data:
        print("No L/R-database receptors found in adata.var_names.")
        empty = pd.DataFrame()
        return empty, empty

    # Every regulon's potential upstream receptors = all receptors present in the data.
    potential = {}
    for reg in regulon_dict:
        tf = reg[:-3] if str(reg).endswith('(+)') else str(reg)
        potential[tf] = list(receptors_in_data)
    if save_to_uns:
        adata.uns['potential_receptor_dict'] = {k: list(v) for k, v in potential.items()}
    print(f"Candidate upstream receptors: {len(receptors_in_data)} (all LR receptors in data); "
          f"{len(potential)} regulons.")

    # ---- columns = all TFs + all pruned targets present in the data ----
    all_cols = sorted(
        (set(potential) | {t for reg in regulon_dict for t in regulon_dict[reg]}) & var
    )
    # involved gene set for the (focused) pairwise computation
    involved = list(dict.fromkeys(receptors_in_data + all_cols))

    # ---- one focused, batch-corrected pairwise computation (shared engine) ----
    lc, lcz = compute_global_spatial_correlation_matrix(
        adata=adata,
        layer_key=layer_key,
        latent_obsm_key=latent_obsm_key,
        batch_key=batch_key,
        n_neighbors=n_neighbors,
        model=model,
        jobs=jobs,
        genes=involved,
        hs=hs,                       # reuse infer()'s graph when provided
        save_path=None,
        cache=False,
    )

    # ---- slice receptors (rows) x {TFs + targets} (cols) ----
    rows = [r for r in receptors_in_data if r in lc.index]
    cols = [c for c in all_cols if c in lc.columns]
    corr = lc.loc[rows, cols].astype(float)
    zsc = lcz.loc[rows, cols].astype(float)

    if save_to_uns:
        adata.uns['receptor_corr'] = corr
        adata.uns['receptor_corr_z'] = zsc
    if save_path:
        corr.to_csv(f"{save_path}_receptor_corr.csv")
        zsc.to_csv(f"{save_path}_receptor_corr_z.csv")
        print(f"Saved receptor-regulon correlations to {save_path}_receptor_corr*.csv")

    print(f"Done: correlation matrix {corr.shape} (receptors x cols).")
    return corr, zsc

# Opt-in print style: no effect unless spagrn.print_style.use('print') is called
from .print_style import wrap_all as _wrap_all
_wrap_all(globals(), ['plot_grn_network', 'plot_receptor_correlation_heatmap', 'plot_receptor_score_distribution',
                      'plot_grn_network_with_common_receptors', 'plot_lr_gene_group_bubble'])
