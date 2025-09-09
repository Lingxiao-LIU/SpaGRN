#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 30 Jun 2024 13:39
# @Author: Yao LI
# @File: SpaGRN/autocor.py
import os
import sys
import time
import warnings  # Added missing import
import scipy
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
from pynndescent import NNDescent
from scipy.stats import chi2, norm
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, issparse
from esda.getisord import G
from esda.moran import Moran
from esda.geary import Geary
import multiprocessing
from tqdm import tqdm
from libpysal import weights


def save_array(array, fn='array.json'):
    import json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    encodedNumpyData = json.dumps(array, cls=NumpyEncoder)
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(encodedNumpyData, f)


def save_list(l, fn='list.txt'):
    with open(fn, 'w') as f:
        f.write('\n'.join(l))


def read_list(fn):
    with open(fn, 'r') as f:
        l = f.read().splitlines()
    return l


def compute_weights(distances, neighborhood_factor=3):
    from math import ceil
    radius_ii = ceil(distances.shape[1] / neighborhood_factor)
    sigma = distances[:, [radius_ii - 1]]
    sigma[sigma == 0] = 1
    weights = np.exp(-1 * distances ** 2 / sigma ** 2)
    wnorm = weights.sum(axis=1, keepdims=True)
    wnorm[wnorm == 0] = 1.0
    weights = weights / wnorm
    return weights


def get_w(neighbors_indices, weights_n):
    """
    Create spatial weights matrix from neighbor indices and weights
    neighbors_indices can be DataFrame or numpy array
    """
    import numpy as np
    from libpysal.weights import W
    import warnings
    
    # Convert DataFrame to numpy array if needed
    if hasattr(neighbors_indices, 'values'):
        neighbors_indices = neighbors_indices.values
    
    n_obs = neighbors_indices.shape[0]
    neighbors_dict = {}
    weights_dict = {}
    
    for i in range(n_obs):
        # Get valid neighbors (exclude -1 and NaN values)
        valid_mask = (~np.isnan(neighbors_indices[i])) & (neighbors_indices[i] >= 0)
        valid_neighbors = neighbors_indices[i][valid_mask].astype(int)
        valid_weights = weights_n.iloc[i][valid_mask].to_numpy() if hasattr(weights_n, 'iloc') else weights_n[i][valid_mask]
        
        if len(valid_neighbors) == 0:
            neighbors_dict[i] = []
            weights_dict[i] = []
        else:
            neighbors_dict[i] = valid_neighbors.tolist()
            weights_dict[i] = valid_weights.tolist()
    
    # Create weights matrix
    w = W(neighbors_dict, weights_dict)
    
    # Validate number of cells
    if w.n != n_obs:
        warnings.warn(f"Weights matrix has {w.n} cells, expected {n_obs}. Adjusting...")
        for i in range(n_obs):
            if i not in neighbors_dict:
                neighbors_dict[i] = []
                weights_dict[i] = []
        w = W(neighbors_dict, weights_dict)
    
    # Check for disconnected components
    if w.n_components > 1:
        warnings.warn(f"Weights matrix has {w.n_components} disconnected components. Consider increasing n_neighbors.")
    
    return w


def flat_weights(cell_names, ind, weights, n_neighbors=30):
    """
    Turn neighbor index into flattened weight matrix
    :param cell_names: Array of cell names
    :param ind: DataFrame or array of neighbor indices
    :param weights: DataFrame or array of weights
    :param n_neighbors: Number of neighbors
    :return: pd.DataFrame with columns 'Cell_x', 'Cell_y', 'Weight'
    """
    cell1 = np.repeat(cell_names, n_neighbors)
    cell2_indices = ind.to_numpy().flatten() if hasattr(ind, 'to_numpy') else ind.flatten()
    valid_mask = cell2_indices != -1  # Check for -1 as integer
    cell1 = cell1[valid_mask]
    cell2 = cell_names[cell2_indices[valid_mask]]
    weight = weights.to_numpy().flatten()[valid_mask] if hasattr(weights, 'to_numpy') else weights.flatten()[valid_mask]
    df = pd.DataFrame({
        "Cell_x": cell1,
        "Cell_y": cell2,
        "Weight": weight
    })
    return df


def square_weights(flat_weights_matrix):
    full_weights_matrix = pd.pivot_table(flat_weights_matrix,
                                        index='Cell_x',
                                        columns='Cell_y',
                                        values='Weight',
                                        fill_value=0)
    return full_weights_matrix


def fdr(ps, axis=0):
    ps = np.asarray(ps)
    ps_in_range = (np.issubdtype(ps.dtype, np.number) and np.all(ps == np.clip(ps, 0, 1)))
    if not ps_in_range:
        raise ValueError("`ps` must include only numbers between 0 and 1.")
    if axis is None:
        axis = 0
        ps = ps.ravel()
    axis = np.asarray(axis)[()]
    if not np.issubdtype(axis.dtype, np.integer) or axis.size != 1:
        raise ValueError("`axis` must be an integer or `None`")
    if ps.size <= 1 or ps.shape[axis] <= 1:
        return ps[()]
    ps = np.moveaxis(ps, axis, -1)
    m = ps.shape[-1]
    order = np.argsort(ps, axis=-1)
    ps = np.take_along_axis(ps, order, axis=-1)
    i = np.arange(1, m + 1)
    ps *= m / i
    np.minimum.accumulate(ps[..., ::-1], out=ps[..., ::-1], axis=-1)
    np.put_along_axis(ps, order, values=ps.copy(), axis=-1)
    ps = np.moveaxis(ps, -1, axis)
    return np.clip(ps, 0, 1)


def cal_s0(w):
    s0 = np.sum(w)
    return s0


def cal_s1(w):
    w_sum = w + w.T
    s1 = 0.5 * np.sum(w_sum ** 2)
    return s1


def cal_s2(w):
    row_sums = np.sum(w, axis=1)
    col_sums = np.sum(w, axis=0)
    total_sums = row_sums + col_sums
    s2 = np.sum(total_sums ** 2)
    return s2


def format_gene_array(gene_array):
    if scipy.sparse.issparse(gene_array):
        gene_array = gene_array.toarray()
    return gene_array.reshape(-1)


def cal_k(gene_expression_matrix: np.ndarray, gene_x_id, n):
    gene_x_exp_mean = gene_expression_matrix[:, gene_x_id].mean()
    gene_x_exp = format_gene_array(gene_expression_matrix[:, gene_x_id])
    denominator = np.square(np.sum(np.square(gene_x_exp - gene_x_exp_mean)))
    numerator = n * np.sum(np.power(gene_x_exp - gene_x_exp_mean, 4))
    K = numerator / denominator
    return K


def somde_p_values(adata, k=20, layer_key='raw_counts', latent_obsm_key="spatial", batch_key=None):
    if layer_key:
        exp = adata.layers[layer_key].toarray() if scipy.sparse.issparse(adata.layers[layer_key]) else adata.layers[layer_key]
    else:
        exp = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    df = pd.DataFrame(data=exp.T, columns=adata.obs_names, index=adata.var_names)
    cell_coordinates = adata.obsm[latent_obsm_key]
    corinfo = pd.DataFrame({"x": cell_coordinates[:, 0], "y": cell_coordinates[:, 1]})
    corinfo.index = adata.obs_names
    corinfo["total_count"] = exp.sum(1)
    
    # Handle batch correction if batch_key is provided
    if batch_key and batch_key in adata.obs.columns:
        # Process each batch separately and combine results
        p_values_list = []
        batches = adata.obs[batch_key].unique()
        
        for batch in batches:
            batch_mask = adata.obs[batch_key] == batch
            batch_corinfo = corinfo.loc[batch_mask]
            batch_df = df.loc[:, batch_mask]
            
            if len(batch_corinfo) < k:
                warnings.warn(f"Batch {batch} has fewer cells ({len(batch_corinfo)}) than k={k}. Skipping this batch.")
                continue
                
            X = batch_corinfo[['x', 'y']].values.astype(np.float32)
            
            try:
                from somde import SomNode
                som = SomNode(X, k)
                som.mtx(batch_df)
                som.norm()
                result, SVnum = som.run()
                p_values_list.append(result.pval)
            except Exception as e:
                warnings.warn(f"SOMDE failed for batch {batch}: {e}")
                continue
        
        if p_values_list:
            # Combine p-values across batches (using Fisher's method or similar)
            from scipy.stats import combine_pvalues
            combined_p_values = []
            for gene_idx in range(len(adata.var_names)):
                gene_p_values = [pvals[gene_idx] for pvals in p_values_list if gene_idx < len(pvals)]
                if gene_p_values:
                    _, combined_p = combine_pvalues(gene_p_values, method='fisher')
                    combined_p_values.append(combined_p)
                else:
                    combined_p_values.append(1.0)  # Default high p-value
            p_values = np.array(combined_p_values)
        else:
            p_values = np.ones(len(adata.var_names))  # Default high p-values
    else:
        X = corinfo[['x', 'y']].values.astype(np.float32)
        from somde import SomNode
        som = SomNode(X, k)
        som.mtx(df)
        som.norm()
        result, SVnum = som.run()
        p_values = result.pval
        som.view()
    
    adjusted_p_values = fdr(p_values)
    return adjusted_p_values


def view(som, raw=True, c=False, line=False):
    import matplotlib.pyplot as plt
    rr = som.som.codebook
    sizenum = np.ones([rr.shape[0], rr.shape[1]]) * 30
    rr = np.reshape(rr, [-1, 2])
    if raw:
        plt.scatter(som.X[:, 0], som.X[:, 1], s=3, label='original')
    for i in range(som.X.shape[0]):
        v, u = som.som.bmus[i]
        sizenum[u, v] += 2
        if line:
            plt.plot([som.X[i, 0], som.som.codebook[u, v, 0]], [som.X[i, 1], som.som.codebook[u, v, 1]])
    sizenum = np.reshape(sizenum, [-1, ])
    if c:
        plt.scatter(rr[:, 0], rr[:, 1], s=sizenum, label=str(som.somn) + 'X' + str(som.somn) + ' SOM nodes', c=sizenum, cmap='hot')
        plt.colorbar()
    else:
        plt.scatter(rr[:, 0], rr[:, 1], s=sizenum, label=str(som.somn) + 'X' + str(som.somn) + ' SOM nodes', c='r')
    plt.savefig('somde.png')
    plt.close()
