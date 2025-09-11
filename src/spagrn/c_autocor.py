#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 31 Dec 2024 15:02
# @Author: Yao LI
# @File: SpaGRN/c_autocor.py
import os
import sys
import time

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


# -----------------------------------------------------#
# G's C
# -----------------------------------------------------#
# def _gearys_c(adata, weights, layer_key='raw_counts'):
#     if 'connectivities' not in adata.obsp.keys():
#         adata.obsp['connectivities'] = weights
#     gearys_c_array = sc.metrics.gearys_c(adata, layer=layer_key)
#     # shape: (n_genes, )
#     return gearys_c_array
#
#
# def _gearys_c_p_value_one_gene(gene_expression_matrix, n_genes, gene_x_id, weights, gearys_c_array):
#     C = gearys_c_array[gene_x_id]
#     EC = 1
#     K = cal_k(gene_expression_matrix, gene_x_id, n_genes)
#     S0 = cal_s0(weights)
#     S1 = cal_s1(weights)
#     S2 = cal_s2(weights)
#     part1 = (n_genes - 1) * S1 * (n_genes ** 2 - 3 * n_genes + 3 - K * (n_genes - 1)) / (np.square(S0) * n_genes * (n_genes - 2) * (n_genes - 3))
#     part2 = (n_genes ** 2 - 3 - K * np.square(n_genes - 1)) / (n_genes * (n_genes - 2) * (n_genes - 3))
#     part3 = (n_genes - 1) * S2 * (n_genes ** 2 + 3 * n_genes - 6 - K * (n_genes ** 2 - n_genes + 2)) / (4 * n_genes * (n_genes - 2) * (n_genes - 3) * np.square(S0))
#     VC = part1 + part2 - part3
#     # variance = (2 * (n ** 2) * S1 - n * S2 + 3 * (S0 ** 2)) / (S0 ** 2 * (n - 1) * (n - 2) * (n - 3))
#     VC_norm = (1 / (2 * (n_genes + 1) * S0 ** 2)) * ((2 * S1 + S2) * (n_genes - 1) - 4 * S0 ** 2)
#     Z = (C - EC) / np.sqrt(VC_norm)
#     p_value = 1 - norm.cdf(Z)
#     print(f'C: {C}\nVC: {VC}\nVC_norm: {VC_norm}\nZ: {Z}\np_value: {p_value}')
#     return p_value


def gearys_c_p_value_one_gene(x, w):
    c = Geary(x, w)
    return c.p_norm


def gearys_c_z_score_one_gene(x, w):
    c = Geary(x, w)
    return c.z_norm


# parallel computing
def _compute_c_for_gene(args):
    x, w = args
    Cp = gearys_c_p_value_one_gene(x, w)
    return Cp


def _gearys_c_parallel(n_genes, gene_expression_matrix, w, n_process=None):
    pool_args = [(gene_expression_matrix[:, gene_x_id], w) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_process) as pool:
        p_values = pool.map(_compute_c_for_gene, pool_args)
    return np.array(p_values)


def _compute_c_z_score_for_gene(args):
    x, w = args
    Cz = gearys_c_z_score_one_gene(x, w)
    return Cz


def _gearys_c_z_score_parallel(n_genes, gene_expression_matrix, w, n_process=None):
    pool_args = [(gene_expression_matrix[:, gene_x_id], w) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_process) as pool:
        z_scores = pool.map(_compute_c_z_score_for_gene, pool_args)
    return np.array(z_scores)


def gearys_c(adata, Weights, layer_key='raw_counts', n_process=None, mode='pvalue', batch_key=None):
    """
    Main function to calculate Geary's C with batch correction
    :param adata: AnnData object
    :param Weights: libpysal.weights.W object or None (if batch_key is used)
    :param layer_key: Layer to use for expression data
    :param n_process: Number of processes for parallel computation
    :param mode: 'pvalue' or 'zscore'
    :param batch_key: Key for batch information in adata.obs
    :return: Array of p-values or z-scores
    """
    import numpy as np
    import pandas as pd
    from esda.geary import Geary
    from libpysal.weights import W
    import warnings
    from scipy.stats import combine_pvalues
    
    # Get expression data
    if layer_key and layer_key in adata.layers:
        exp_data = adata.layers[layer_key]
        if hasattr(exp_data, 'toarray'):
            exp_data = exp_data.toarray()
    else:
        exp_data = adata.X
        if hasattr(exp_data, 'toarray'):
            exp_data = exp_data.toarray()
    
    n_genes = exp_data.shape[1]
    n_cells = exp_data.shape[0]
    
    if batch_key and batch_key in adata.obs.columns:
        # Batch-aware processing
        batches = adata.obs[batch_key].unique()
        batch_results = []
        
        for batch in batches:
            batch_mask = adata.obs[batch_key] == batch
            batch_indices = np.where(batch_mask)[0]
            
            if len(batch_indices) < 5:
                warnings.warn(f"Batch {batch} has too few cells ({len(batch_indices)}), skipping")
                continue
            
            # Create batch-specific weights
            subset_ids = batch_indices
            subset_neighbors = {}
            subset_weights_dict = {}
            subset_set = set(subset_ids)
            id_map = {old_id: new_id for new_id, old_id in enumerate(subset_ids)}
            
            for old_id in subset_ids:
                old_neighbors = Weights.neighbors.get(old_id, [])
                new_neighbors = [id_map[j] for j in old_neighbors if j in subset_set]
                old_w = Weights.weights.get(old_id, [])
                new_w = [old_w[idx] for idx, j in enumerate(old_neighbors) if j in subset_set]
                subset_neighbors[id_map[old_id]] = new_neighbors
                subset_weights_dict[id_map[old_id]] = new_w
            
            batch_weights = W(subset_neighbors, subset_weights_dict)
            
            if batch_weights.n == 0:
                warnings.warn(f"No valid weights for batch {batch}, skipping")
                continue
            
            if batch_weights.n != len(batch_indices):
                warnings.warn(f"Batch {batch} weights matrix has {batch_weights.n} cells, expected {len(batch_indices)}")
                continue
            
            batch_exp = exp_data[batch_indices, :]
            batch_results_batch = []
            
            for gene_idx in range(n_genes):
                gene_exp = batch_exp[:, gene_idx]
                
                if np.var(gene_exp) == 0 or np.all(gene_exp == 0):
                    batch_results_batch.append(1.0 if mode == 'pvalue' else 0.0)
                    continue
                
                try:
                    c_stat = Geary(gene_exp, batch_weights)
                    if mode == 'pvalue':
                        pval = c_stat.p_norm if hasattr(c_stat, 'p_norm') else 1.0
                    else:  # zscore
                        pval = c_stat.z_norm if hasattr(c_stat, 'z_norm') else 0.0
                    
                    if np.isnan(pval) or np.isinf(pval):
                        pval = 1.0 if mode == 'pvalue' else 0.0
                    elif mode == 'pvalue' and (pval < 0 or pval > 1):
                        pval = np.clip(pval, 0, 1)
                    
                    batch_results_batch.append(pval)
                    
                except Exception as e:
                    warnings.warn(f"Error computing Geary's C for gene {adata.var_names[gene_idx]} in batch {batch}: {e}")
                    batch_results_batch.append(1.0 if mode == 'pvalue' else 0.0)
            
            batch_results.append(np.array(batch_results_batch))
        
        if not batch_results:
            warnings.warn("No valid batches found, returning default values")
            return np.ones(n_genes) if mode == 'pvalue' else np.zeros(n_genes)
        
        if mode == 'pvalue':
            combined_pvalues = []
            for gene_idx in range(n_genes):
                gene_pvals = [batch_result[gene_idx] for batch_result in batch_results 
                             if gene_idx < len(batch_result)]
                if gene_pvals:
                    try:
                        _, combined_p = combine_pvalues(gene_pvals, method='fisher')
                        combined_pvalues.append(combined_p)
                    except:
                        combined_pvalues.append(1.0)
                else:
                    combined_pvalues.append(1.0)
            return np.array(combined_pvalues)
        else:
            combined_zscores = []
            for gene_idx in range(n_genes):
                gene_zscores = [batch_result[gene_idx] for batch_result in batch_results 
                               if gene_idx < len(batch_result)]
                if gene_zscores:
                    combined_zscores.append(np.mean(gene_zscores))
                else:
                    combined_zscores.append(0.0)
            return np.array(combined_zscores)
    
    else:
        if Weights.n != n_cells:
            raise ValueError(f"Weights matrix has {Weights.n} cells, but expression data has {n_cells} cells")
        
        p_values = _gearys_c_parallel(n_genes, exp_data, Weights, n_process=n_process)
        return p_values