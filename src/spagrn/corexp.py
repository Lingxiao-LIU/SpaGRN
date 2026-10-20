#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yao LI
# @File: SpaGRN/corexp.py
"""
Global bivariate spatial correlation (Moran's R / Geary's C) for TF-target pairs.

Batch-corrected: each gene is centred on its own sample's mean (within-sample
centring) and the denominator is the within-sample sum of squares. Because the
spatial weights are batch-aware (block-diagonal, no cross-sample edges), every
edge in the numerator is within a single sample, so cells from different samples
are never correlated across the batch boundary.

Performance: the gene x cell expression matrix and the per-edge index/weight
vectors are written once to a memory-mapped temp file and read read-only by the
workers, so they are never pickled per TF-target pair. Per-gene per-sample means
and denominators are precomputed once instead of being recomputed per pair.
"""

import os
import time
import tempfile
import shutil

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
import scipy.sparse as sp
from joblib import Parallel, delayed, dump, load
from tqdm.auto import tqdm


def _precompute_gene_stats(layer, batches):
    """Per-gene per-sample means, per-cell sample codes, and within-sample denominators.

    Returns
    -------
    batch_means : (n_batches, n_genes) within-sample mean of each gene
    codes       : (n_cells,) int       sample code per cell
    denom       : (n_genes,)           sqrt of within-sample sum of squares
    """
    issp = issparse(layer)
    n_cells, n_genes = layer.shape
    layer = layer.tocsr() if issp else np.asarray(layer)

    def _csum(m):
        return np.asarray(m.sum(axis=0)).ravel()

    def _csq(m):
        return np.asarray(m.multiply(m).sum(axis=0)).ravel() if issparse(m) else (np.asarray(m) ** 2).sum(axis=0)

    if batches is None:
        codes = np.zeros(n_cells, dtype=np.int64)
        s = _csum(layer)
        bm = s / n_cells
        sos = _csq(layer) - 2.0 * bm * s + (bm ** 2) * n_cells
        return bm[None, :].astype(np.float64), codes, np.sqrt(np.maximum(sos, 0.0)).astype(np.float64)

    uniq = pd.unique(batches)
    codes = np.empty(n_cells, dtype=np.int64)
    batch_means = np.zeros((len(uniq), n_genes))
    sos = np.zeros(n_genes)
    for bi, b in enumerate(uniq):
        mask = (batches == b)
        codes[mask] = bi
        nb = int(mask.sum())
        if nb == 0:
            continue
        sub = layer[mask]
        s = _csum(sub)
        bm = s / nb
        batch_means[bi] = bm
        # within-sample sum of squares: sum_c (x_c - bm)^2
        sos += _csq(sub) - 2.0 * bm * s + (bm ** 2) * nb

    return batch_means.astype(np.float64), codes, np.sqrt(np.maximum(sos, 0.0)).astype(np.float64)


def _pair_moran(gx, gy, mtxT, cell_x_id, cell_y_id, wij, bmeansT, code_x, code_y, denom):
    d = denom[gx] * denom[gy]
    if d == 0:
        return gx, gy, 0.0
    x = mtxT[gx]
    y = mtxT[gy]
    cx = x[cell_x_id] - bmeansT[gx][code_x]   # within-sample centred at edge endpoints
    cy = y[cell_y_id] - bmeansT[gy][code_y]
    return gx, gy, float(np.dot(wij, cx * cy) / d)


def _pair_geary(gx, gy, mtxT, cell_x_id, cell_y_id, wij, bmeansT, code_x, code_y, denom):
    d = denom[gx] * denom[gy]
    if d == 0:
        return gx, gy, 0.0
    x = mtxT[gx]
    y = mtxT[gy]
    # within-sample centring cancels in the within-edge differences
    num = np.dot(wij, (x[cell_x_id] - x[cell_y_id]) * (y[cell_x_id] - y[cell_y_id]))
    return gx, gy, float(num / d)


def _build_sparse_weights(adata, weights, n):
    """Build an (n x n) sparse weight matrix W[i,j]=w_ij from the flat edge list.

    Maps cell names -> integer positions via a dict (NOT .loc on a 90M-element
    string list, which materialises an ~8 GB <U24 array and OOMs at scale).
    """
    name_to_id = {name: i for i, name in enumerate(adata.obs_names)}
    cx = weights['Cell_x'].map(name_to_id).to_numpy(dtype=np.int64)
    cy = weights['Cell_y'].map(name_to_id).to_numpy(dtype=np.int64)
    wij = weights['Weight'].to_numpy(dtype=np.float64)
    W = sp.csr_matrix((wij, (cx, cy)), shape=(n, n))
    del cx, cy, wij
    return W


def _bivariate_driver(adata, weights, tfs_in_data, select_genes, layer_key, num_workers, batch_key, kind):
    """Vectorised global bivariate Moran's R / Geary's C for all TF x target pairs.

    Mathematically identical to the previous per-edge implementation (validated to
    floating-point tolerance), but expressed as sparse-matrix products: the whole
    TF x target matrix is computed with one sparse-dense matmul (plus a few dense
    matmuls for Geary), instead of looping over millions of edges per pair. This
    also removes the per-edge index arrays entirely, so it no longer allocates the
    ~8 GB string arrays that OOM at 900k cells.
    """
    gene_names = adata.var.index
    tf_ids = gene_names.get_indexer(tfs_in_data)
    target_ids = gene_names.get_indexer(select_genes)
    n = adata.n_obs

    layer = adata.layers[layer_key] if layer_key else adata.X
    batches = None
    if batch_key and batch_key in adata.obs.columns:
        batches = adata.obs[batch_key].values
        print(f"Using batch correction with batch key: {batch_key}")

    # Per-gene per-sample means + within-sample denominators (sqrt of within-sample SoS).
    batch_means, codes, denom = _precompute_gene_stats(layer, batches)

    # Keep expression sparse (CSC for cheap column slicing). We NEVER densify the
    # full (cells x genes) matrix -- only per-chunk column blocks -- so peak memory
    # is bounded by one chunk regardless of cell/gene count.
    Lsp = layer.tocsc() if issparse(layer) else csr_matrix(np.asarray(layer)).tocsc()

    # Sparse weights matrix (no per-edge string materialisation).
    W = _build_sparse_weights(adata, weights, n)

    start = time.time()
    dtf = denom[tf_ids]
    n_tf, n_tar = len(tf_ids), len(target_ids)
    TARGET_CHUNK = 2000
    R = np.zeros((n_tf, n_tar))
    print(f"bivariate {kind}: {n_tf} TFs x {n_tar} targets "
          f"({n_tf * n_tar} pairs) via sparse matmul (chunked, low-mem)")

    def _dense_cols(col_ids):
        return np.ascontiguousarray(Lsp[:, col_ids].toarray(), dtype=np.float64)

    def _center_cols(block, col_ids):
        # within-sample centring: subtract each cell's sample mean for these genes
        return block - batch_means[np.ix_(codes, col_ids)]

    if kind == 'moran':
        ZX = _center_cols(_dense_cols(tf_ids), tf_ids)        # cells x n_tf (small)
        for c0 in range(0, n_tar, TARGET_CHUNK):
            c1 = min(c0 + TARGET_CHUNK, n_tar)
            tcols = target_ids[c0:c1]
            ZY = _center_cols(_dense_cols(tcols), tcols)      # cells x chunk
            Num = ZX.T @ (W @ ZY)                             # n_tf x chunk
            dd = np.outer(dtf, denom[tcols])
            R[:, c0:c1] = np.divide(Num, dd, out=np.zeros_like(Num), where=dd != 0)
            del ZY
    else:
        rs = np.asarray(W.sum(1)).ravel()
        cs = np.asarray(W.sum(0)).ravel()
        Xt = _dense_cols(tf_ids)                              # cells x n_tf (raw)
        WXt = W @ Xt
        Xt_rs = Xt * rs[:, None]
        Xt_cs = Xt * cs[:, None]
        for c0 in range(0, n_tar, TARGET_CHUNK):
            c1 = min(c0 + TARGET_CHUNK, n_tar)
            tcols = target_ids[c0:c1]
            Yt = _dense_cols(tcols)                           # cells x chunk (raw)
            Num = (Xt_rs.T @ Yt) - (Xt.T @ (W @ Yt)) - (WXt.T @ Yt) + (Xt_cs.T @ Yt)
            dd = np.outer(dtf, denom[tcols])
            R[:, c0:c1] = np.divide(Num, dd, out=np.zeros_like(Num), where=dd != 0)
            del Yt

    # Long-form output (TF, target, importance), same columns as before.
    tf_names = gene_names[tf_ids].to_numpy()
    tar_names = gene_names[target_ids].to_numpy()
    out = pd.DataFrame({
        'TF': np.repeat(tf_names, len(target_ids)),
        'target': np.tile(tar_names, len(tf_ids)),
        'importance': R.ravel(),
    })
    print(f"global_bivariate_{kind}: {time.time() - start:.2f}s "
          f"({len(tf_ids) * len(target_ids)} pairs)")
    return out


def global_bivariate_moran_R(adata, weights: pd.DataFrame, tfs_in_data: list, select_genes: list,
                             layer_key='raw_counts', num_workers: int = 4, batch_key=None) -> pd.DataFrame:
    """
    Global bivariate Moran's R for every TF x target pair, with within-sample
    batch correction when batch_key is given.

    Parameters
    ----------
    adata : anndata.AnnData
    weights : pd.DataFrame      flattened spatial weights (columns Cell_x, Cell_y, Weight)
    tfs_in_data : list          transcription factors
    select_genes : list         target genes
    layer_key : str             expression layer
    num_workers : int           parallel workers
    batch_key : str, optional   adata.obs column with sample labels

    Returns
    -------
    pd.DataFrame with columns TF, target, importance
    """
    return _bivariate_driver(adata, weights, tfs_in_data, select_genes,
                             layer_key, num_workers, batch_key, kind='moran')


def global_bivariate_gearys_C(adata, weights: pd.DataFrame, tfs_in_data: list, select_genes: list,
                              layer_key='raw_counts', num_workers: int = 4, batch_key=None) -> pd.DataFrame:
    """
    Global bivariate Geary's C for every TF x target pair, with within-sample
    batch correction when batch_key is given. Parameters as in global_bivariate_moran_R.
    """
    return _bivariate_driver(adata, weights, tfs_in_data, select_genes,
                             layer_key, num_workers, batch_key, kind='geary')