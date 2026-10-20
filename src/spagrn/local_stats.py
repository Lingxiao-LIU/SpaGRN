import os
import sys
import numpy as np
from numba import jit
from tqdm import tqdm
import pandas as pd
from scipy.stats import norm
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
import multiprocessing

from . import danb_model
from . import bernoulli_model
from . import normal_model
from . import none_model

from .knn import compute_node_degree
from .utils import center_values


@jit(nopython=True)
def local_cov_weights(x, neighbors, weights):
    out = 0

    for i in range(len(x)):
        for k in range(neighbors.shape[1]):

            j = neighbors[i, k]
            w_ij = weights[i, k]

            xi = x[i]
            xj = x[j]
            if xi == 0 or xj == 0 or w_ij == 0:
                out += 0
            else:
                out += xi * xj * w_ij

    return out


@jit(nopython=True)
def compute_moments_weights_slow(mu, x2, neighbors, weights):
    """
    This version exhaustively iterates over all |E|^2 terms
    to compute the expected moments exactly.  Used to test
    the more optimized formulations that follow.
    """

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            EG += wij * mu[i] * mu[j]

    # Calculate E[G^2]
    EG2 = 0
    for i in range(N):

        EG2_i = 0

        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            for x in range(N):
                for z in range(K):
                    y = neighbors[x, z]
                    wxy = weights[x, z]

                    s = wij * wxy
                    if s == 0:
                        continue

                    if i == x:
                        if j == y:
                            t1 = x2[i] * x2[j]
                        else:
                            t1 = x2[i] * mu[j] * mu[y]
                    elif i == y:
                        if j == x:
                            t1 = x2[i] * x2[j]
                        else:
                            t1 = x2[i] * mu[j] * mu[x]
                    else:  # i is unique since i can't equal j

                        if j == x:
                            t1 = mu[i] * x2[j] * mu[y]
                        elif j == y:
                            t1 = mu[i] * x2[j] * mu[x]
                        else:  # i and j are unique, no shared nodes
                            t1 = mu[i] * mu[j] * mu[x] * mu[y]

                    EG2_i += s * t1

        EG2 += EG2_i

    return EG, EG2


@jit(nopython=True)
def compute_moments_weights(mu, x2, neighbors, weights):

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            EG += wij * mu[i] * mu[j]

    # Calculate E[G^2]
    EG2 = 0

    #   Get the x^2*y*z terms
    t1 = np.zeros(N)
    t2 = np.zeros(N)

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]
            if wij == 0:
                continue

            t1[i] += wij * mu[j]
            t2[i] += wij**2 * mu[j] ** 2

            t1[j] += wij * mu[i]
            t2[j] += wij**2 * mu[i] ** 2

    t1 = t1**2

    for i in range(N):
        EG2 += (x2[i] - mu[i] ** 2) * (t1[i] - t2[i])

    #  Get the x^2*y^2 terms
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]

            EG2 += wij**2 * (x2[i] * x2[j] - (mu[i] ** 2) * (mu[j] ** 2))

    EG2 += EG**2

    return EG, EG2


@jit(nopython=True)
def compute_local_cov_max(node_degrees, vals):
    tot = 0.0

    for i in range(node_degrees.size):
        tot += node_degrees[i] * (vals[i] ** 2)

    return tot / 2


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _job_wrapper(args):
    """Top-level wrapper so joblib can pickle it cleanly."""
    vals, neighbors, weights, num_umi, model, centered, Wtot2, D, batches = args
    return _compute_hs_inner(vals, neighbors, weights, num_umi, model, centered, Wtot2, D, batches)


def compute_hs(
    counts, neighbors, weights, num_umi, model, genes,
    centered=False, jobs=1, batches=None
):
    """Compute Hotspot autocorrelation scores.

    Uses joblib with the 'loky' backend for parallelism, which works correctly
    on Windows, Linux, and macOS regardless of array size. Large shared arrays
    (neighbors, weights, num_umi, D) are passed via joblib's memory-mapping
    so they are never fully copied into each worker process.
    """
    from joblib import Parallel, delayed, dump, load
    import tempfile

    neighbors_arr = np.ascontiguousarray(neighbors.values).astype(np.int64)
    weights_arr   = np.ascontiguousarray(weights.values).astype(np.float64)
    num_umi_arr   = np.ascontiguousarray(num_umi.values).astype(np.float64)

    D     = compute_node_degree(neighbors_arr, weights_arr)
    Wtot2 = (weights_arr ** 2).sum()

    # Pre-extract all gene vectors into a list (sparse-safe)
    n_genes = counts.shape[0]
    all_vals = []
    for i in range(n_genes):
        v = counts[i]
        if issparse(v):
            v = v.toarray().ravel()
        all_vals.append(v.astype("double"))

    if jobs > 1:
        # ------------------------------------------------------------------
        # Write large shared arrays to a memory-mapped temp file.
        # joblib workers will load them as read-only mmaps — zero copy cost,
        # works on all platforms, no pipe-size limits.
        # ------------------------------------------------------------------
        tmp_dir = tempfile.mkdtemp(prefix="spagrn_hs_")
        try:
            mmap_path = os.path.join(tmp_dir, "shared_arrays.pkl")
            dump(
                (neighbors_arr, weights_arr, num_umi_arr, D, Wtot2, model, centered, batches),
                mmap_path,
                compress=False,
            )
            shared = load(mmap_path, mmap_mode="r")
            nbr_m, wgt_m, umi_m, D_m, Wtot2_m, model_m, centered_m, batches_m = shared

            results = Parallel(n_jobs=jobs, backend="loky", verbose=0)(
                delayed(_compute_hs_inner)(
                    vals, nbr_m, wgt_m, umi_m,
                    model_m, centered_m, Wtot2_m, D_m, batches_m
                )
                for vals in tqdm(all_vals, total=n_genes)
            )
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        results = [
            _compute_hs_inner(
                vals, neighbors_arr, weights_arr, num_umi_arr,
                model, centered, Wtot2, D, batches
            )
            for vals in tqdm(all_vals, total=n_genes)
        ]

    results = pd.DataFrame(results, index=genes, columns=["G", "EG", "stdG", "Z", "C"])

    results["Pval"] = norm.sf(results["Z"].values)
    results["FDR"]  = multipletests(results["Pval"], method="fdr_bh")[1]

    results = results.sort_values("Z", ascending=False)
    results.index.name = "Gene"

    return results[["C", "Z", "Pval", "FDR"]]


# ---------------------------------------------------------------------------
# Inner computation (called by both single- and multi-process paths)
# ---------------------------------------------------------------------------

def _compute_hs_inner(vals, neighbors, weights, num_umi, model, centered, Wtot2, D, batches=None):
    """Compute Hotspot statistics for a single gene."""

    if model == "bernoulli":
        vals = (vals > 0).astype("double")
        mu, var, x2 = bernoulli_model.fit_gene_model(vals, num_umi)
    elif model == "danb":
        mu, var, x2 = danb_model.fit_gene_model(vals, num_umi)
    elif model == "normal":
        mu, var, x2 = normal_model.fit_gene_model(vals, num_umi)
    elif model == "none":
        mu, var, x2 = none_model.fit_gene_model(vals, num_umi)
    else:
        raise Exception("Invalid Model: {}".format(model))

    # Apply batch correction if batches are provided
    if batches is not None and centered:
        vals = apply_batch_correction(vals, mu, var, batches)
    elif centered:
        vals = center_values(vals, mu, var)

    G = local_cov_weights(vals, neighbors, weights)

    if centered:
        EG, EG2 = 0, Wtot2
    else:
        EG, EG2 = compute_moments_weights(mu, x2, neighbors, weights)

    stdG = (EG2 - EG * EG) ** 0.5

    Z = (G - EG) / stdG

    G_max = compute_local_cov_max(D, vals)
    C = (G - EG) / G_max

    return [G, EG, stdG, Z, C]


def apply_batch_correction(vals, mu, var, batches):
    """Center expression values within each batch."""
    if batches is None:
        return center_values(vals, mu, var)

    corrected_vals = np.zeros_like(vals)
    for batch in np.unique(batches):
        mask = batches == batch
        if mask.sum() == 0:
            continue
        if mask.sum() == 1:
            corrected_vals[mask] = 0.0
            continue
        corrected_vals[mask] = center_values(vals[mask], mu[mask], var[mask])

    return corrected_vals