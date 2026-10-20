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
from scipy.sparse import csr_matrix, issparse, diags
from esda.getisord import G
from esda.moran import Moran
from esda.geary import Geary
import multiprocessing
from tqdm.auto import tqdm
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
    :param ind: DataFrame or array of neighbor indices (can be integer indices or cell names)
    :param weights: DataFrame or array of weights
    :param n_neighbors: Number of neighbors
    :return: pd.DataFrame with columns 'Cell_x', 'Cell_y', 'Weight'
    """
    import numpy as np
    import pandas as pd
    
    # Convert to numpy arrays if DataFrames
    if hasattr(ind, 'values'):
        ind_array = ind.values
    else:
        ind_array = ind
    
    if hasattr(weights, 'values'):
        weights_array = weights.values
    else:
        weights_array = weights
    
    # Convert cell_names to numpy array
    cell_names = np.asarray(cell_names)
    
    # Create a mapping from cell names to indices
    name_to_idx = {name: idx for idx, name in enumerate(cell_names)}
    
    # Check if ind contains cell names (strings) or indices (integers)
    first_valid = None
    for val in ind_array.flat:
        if pd.notna(val) and val != -1:
            first_valid = val
            break
    
    # Determine if we have cell names or indices
    if first_valid is not None and isinstance(first_valid, str):
        # ind contains cell names, convert to indices
        vectorized_map = np.vectorize(lambda x: name_to_idx.get(x, -1) if pd.notna(x) else -1)
        ind_array = vectorized_map(ind_array)
    
    # Now ind_array should contain integer indices
    # Flatten and repeat cell names
    cell1 = np.repeat(cell_names, n_neighbors)
    cell2_indices = ind_array.flatten()
    
    # Create valid mask - must be integers and within bounds
    valid_mask = (cell2_indices != -1) & (cell2_indices >= 0) & (cell2_indices < len(cell_names))
    
    # Filter arrays
    cell1 = cell1[valid_mask]
    cell2_indices_valid = cell2_indices[valid_mask].astype(int)
    cell2 = cell_names[cell2_indices_valid]
    weight = weights_array.flatten()[valid_mask]
    
    df = pd.DataFrame({
        "Cell_x": cell1,
        "Cell_y": cell2,
        "Weight": weight
    })
    
    return df


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

# ===========================================================================
# Unified, batch-corrected spatial autocorrelation
# (folded in from the former batch_autocor module)
#
# Moran's I, Geary's C and Getis-Ord G are computed the same way: each sample is
# scored independently (within-sample neighbours and within-sample centring) and
# the per-sample results are combined across samples - Fisher for p-values, mean
# for z-scores. All three statistics share one parallel pass and one set of
# per-sample weights. esda is called with permutations=0 (p_norm / z_norm are
# analytic), and the expression matrix is never fully densified.
# ===========================================================================

def build_batch_weights(Weights, batch_labels, min_cells=5):
    """Split a (block-diagonal) global weights matrix into per-sample sub-weights.

    Returns a list of (W_batch, global_index_array), built once and reused by
    every statistic and by both p-value and z-score modes.
    """
    from libpysal.weights import W

    batch_labels = np.asarray(batch_labels)
    out = []
    for b in pd.unique(batch_labels):
        idx = np.where(batch_labels == b)[0]
        if idx.size < min_cells:
            warnings.warn(f"Batch '{b}': only {idx.size} cells (<{min_cells}); skipping.")
            continue

        id_map = {int(old): new for new, old in enumerate(idx)}
        nbrs, wts = {}, {}
        for old in idx:
            old = int(old)
            on = Weights.neighbors.get(old, [])
            ow = Weights.weights.get(old, [])
            nn, nw = [], []
            for k, j in enumerate(on):
                nj = id_map.get(int(j))
                if nj is not None:        # drops any cross-sample edge (there should be none)
                    nn.append(nj)
                    nw.append(ow[k])
            nbrs[id_map[old]] = nn
            wts[id_map[old]] = nw

        out.append((W(nbrs, wts, silence_warnings=True), idx))
    return out


def _safe(fn, default):
    try:
        v = fn()
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _row_standardize(Wm):
    """Row-standardize a sparse weights matrix (esda transformation='R')."""
    rs = np.asarray(Wm.sum(1)).ravel()
    inv = np.divide(1.0, rs, out=np.zeros_like(rs, dtype=float), where=rs > 0)
    return diags(inv) @ Wm


def _weight_moments(Wm):
    """s0, s1, s2 exactly as libpysal/esda compute them for a given transform."""
    s0 = float(Wm.sum())
    WmT = Wm.T.tocsr()
    s1 = 0.5 * float(((Wm + WmT).power(2)).sum())
    rowsum = np.asarray(Wm.sum(1)).ravel()
    colsum = np.asarray(Wm.sum(0)).ravel()
    s2 = float(((rowsum + colsum) ** 2).sum())
    return s0, s1, s2, rowsum, colsum


def _compute_batch_stats(task):
    """Compute Moran's I, Geary's C, Getis-Ord G for every gene in one sample.

    Vectorised, exact reimplementation of the esda analytic (permutations=0)
    z_norm / p_norm statistics — validated to floating-point tolerance against
    esda.Moran / esda.Geary / esda.getisord.G (transform 'R','R','B' respectively).
    Genes are processed in chunks to bound peak memory from the dense centring.
    """
    batch_expr, W_b, mode = task
    is_p = (mode == "pvalue")
    default = 1.0 if is_p else 0.0

    # raw (original-transform) sparse weights for this sample
    if hasattr(W_b, "sparse"):
        try:
            W_b.transform = 'O'
        except Exception:
            pass
        Wo = W_b.sparse.tocsr().astype(float)
    elif issparse(W_b):
        Wo = W_b.tocsr().astype(float)
    else:
        Wo = csr_matrix(np.asarray(W_b, dtype=float))

    n = Wo.shape[0]
    Xsp = batch_expr.tocsr() if issparse(batch_expr) else csr_matrix(np.asarray(batch_expr, dtype=float))
    n_genes = Xsp.shape[1]

    I = np.full(n_genes, default)
    C = np.full(n_genes, default)
    Gg = np.full(n_genes, default)

    if n < 4:
        # esda G variance denominator uses (n-2)(n-3); too few cells -> defaults
        return I, C, Gg

    # ---- weight-only moments (independent of gene), computed once ----
    Wr = _row_standardize(Wo)                       # Moran / Geary (transform 'R')
    s0r, s1r, s2r, rs_r, cs_r = _weight_moments(Wr)
    Wb = Wo.copy(); Wb.data[:] = 1.0                # Getis-Ord G (transform 'B')
    Wb.eliminate_zeros()
    s0b, s1b, s2b, _, _ = _weight_moments(Wb)

    n2 = n * n
    # Moran analytic moments
    EI = -1.0 / (n - 1)
    VI = (n2 * s1r - n * s2r + 3 * s0r * s0r) / (s0r * s0r * (n2 - 1)) - EI * EI
    # Geary analytic moments
    EC = 1.0
    VC = (1.0 / (2 * (n + 1) * s0r * s0r)) * ((2 * s1r + s2r) * (n - 1) - 4 * s0r * s0r)
    # G analytic moments (b-coefficients depend only on weights/n)
    s02 = s0b * s0b
    EG = s0b / (n * (n - 1))
    b0 = (n2 - 3 * n + 3) * s1b - n * s2b + 3 * s02
    b1 = -1.0 * ((n2 - n) * s1b - 2 * n * s2b + 6 * s02)
    b2 = -1.0 * (2 * n * s1b - (n + 3) * s2b + 6 * s02)
    b3 = 4 * (n - 1) * s1b - 2 * (n + 1) * s2b + 8 * s02
    b4 = s1b - s2b + s02

    sqrtVI = np.sqrt(VI) if VI > 0 else np.nan
    sqrtVC = np.sqrt(VC) if VC > 0 else np.nan

    CHUNK = 256
    for c0 in range(0, n_genes, CHUNK):
        c1 = min(c0 + CHUNK, n_genes)
        Xd = Xsp[:, c0:c1].toarray()                # n x chunk (dense; the memory driver)
        col_mean = Xd.mean(0)
        Zc = Xd - col_mean
        z2 = Zc * Zc
        den = z2.sum(0)                              # == n*var; 0 iff zero-variance gene
        nonzero = den > 0

        with np.errstate(divide='ignore', invalid='ignore'):
            # ---- Moran's I (transform 'R') ----
            num_I = (Zc * (Wr @ Zc)).sum(0)
            Ival = (n / s0r) * num_I / den
            zI = (Ival - EI) / sqrtVI

            # ---- Geary's C (transform 'R') ----
            num_C = (rs_r @ z2) + (cs_r @ z2) - 2.0 * (Zc * (Wr @ Zc)).sum(0)
            Cval = ((n - 1) / (2.0 * s0r)) * num_C / den
            zC = (Cval - EC) / sqrtVC

            # ---- Getis-Ord G (transform 'B', raw positive values) ----
            Sy = Xd.sum(0); Sy2 = (Xd * Xd).sum(0)
            Sy3 = (Xd ** 3).sum(0); Sy4 = (Xd ** 4).sum(0)
            denG = Sy * Sy - Sy2
            num_G = (Xd * (Wb @ Xd)).sum(0)
            Gval = num_G / denG
            EG2 = b0 * Sy2 ** 2 + b1 * Sy4 + b2 * (Sy ** 2) * Sy2 + b3 * Sy * Sy3 + b4 * Sy ** 4
            EG2DEN = (denG ** 2) * n * (n - 1) * (n - 2) * (n - 3)
            VG = EG2 / EG2DEN - EG * EG
            zG = (Gval - EG) / np.sqrt(VG)

        if is_p:
            # esda p_norm conventions: Moran is TWO-tailed; Geary and G are one-tailed
            pI = 2.0 * (1.0 - norm.cdf(np.abs(zI)))
            pC = 1.0 - norm.cdf(np.abs(zC))
            pG = 1.0 - norm.cdf(np.abs(zG))
            outI, outC, outG = pI, pC, pG
        else:
            outI, outC, outG = zI, zC, zG

        # finite + zero-variance guards -> default (matches the old _safe behaviour)
        def _finalize(vals, ok):
            vals = np.where(np.isfinite(vals) & ok, vals, default)
            return vals
        I[c0:c1]  = _finalize(outI, nonzero)
        C[c0:c1]  = _finalize(outC, nonzero)
        Gg[c0:c1] = _finalize(outG, nonzero & np.isfinite(denG) & (denG != 0))

    return I, C, Gg


def _combine_pvalues_fisher(p_stack):
    """Fisher's method, vectorised across genes. p_stack: (n_samples, n_genes)."""
    p = np.clip(p_stack, 1e-300, 1.0)
    stat = -2.0 * np.log(p).sum(axis=0)
    df = 2 * p.shape[0]
    return chi2.sf(stat, df)


def compute_autocorrelation_stats(adata, Weights, batch_weights=None,
                                  layer_key="raw_counts", mode="pvalue",
                                  n_process=None, batch_key=None):
    """Batch-corrected Moran's I, Geary's C and Getis-Ord G in a single pass.

    Returns a dict with keys 'I', 'C', 'G', each aligned to adata.var_names.
    """
    M = adata.layers[layer_key] if (layer_key and layer_key in adata.layers) else adata.X
    M = M.tocsr() if issparse(M) else np.asarray(M)
    n_genes = M.shape[1]

    if batch_weights is None:
        if batch_key and batch_key in adata.obs.columns:
            batch_weights = build_batch_weights(Weights, adata.obs[batch_key].values)
        else:
            batch_weights = [(Weights, np.arange(adata.n_obs))]

    if not batch_weights:
        warnings.warn("No valid samples; returning default values.")
        default = np.ones(n_genes) if mode == "pvalue" else np.zeros(n_genes)
        return {"I": default.copy(), "C": default.copy(), "G": default.copy()}

    tasks = [(M[idx, :], W_b, mode) for (W_b, idx) in batch_weights]

    desc = "Moran/Geary/Getis (per-sample)"
    if n_process is not None and n_process <= 1:
        results = [_compute_batch_stats(t) for t in tqdm(tasks, desc=desc, unit="sample")]
    else:
        # imap_unordered: results stream back as samples finish; combination is
        # order-independent (Fisher / mean across samples), so order does not matter.
        with multiprocessing.Pool(processes=n_process) as pool:
            results = list(tqdm(pool.imap_unordered(_compute_batch_stats, tasks),
                                total=len(tasks), desc=desc, unit="sample"))

    I_s = np.vstack([r[0] for r in results])
    C_s = np.vstack([r[1] for r in results])
    G_s = np.vstack([r[2] for r in results])

    if mode == "pvalue":
        return {
            "I": _combine_pvalues_fisher(I_s),
            "C": _combine_pvalues_fisher(C_s),
            "G": _combine_pvalues_fisher(G_s),
        }
    # Mean across samples (matches existing z-score combination).
    # For Stouffer instead: I_s.sum(0) / np.sqrt(I_s.shape[0]).
    return {"I": I_s.mean(axis=0), "C": C_s.mean(axis=0), "G": G_s.mean(axis=0)}