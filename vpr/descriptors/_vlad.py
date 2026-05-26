"""Shared VLAD aggregation: a set of local descriptors -> one global vector.

Standard VLAD with intra-normalisation + signed-sqrt power-law + global L2 —
based on NetVLAD-family — applied to a DL extractor's local descriptors so
the resulting global vector matches what ``loop_fusion``'s C++ VladCodebook
produces at runtime (DL-VINS/src/loop_fusion/src/vlad.cpp).
"""
from __future__ import annotations

import numpy as np


def build_codebook(descs: np.ndarray, n_clusters: int, seed: int = 0):
    """k-means a VLAD codebook from a pool of local descriptors."""
    from sklearn.cluster import MiniBatchKMeans

    k = max(1, min(n_clusters, len(descs)))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3)
    km.fit(descs.astype(np.float32))
    return km


def vlad_encode(local: np.ndarray, km, n_clusters: int) -> np.ndarray:
    """local: (K, D) local descriptors -> (n_clusters*D,) VLAD vector."""
    d = int(km.cluster_centers_.shape[1])
    vlad = np.zeros((n_clusters, d), dtype=np.float32)
    if len(local):
        local = local.astype(np.float32)
        words = km.predict(local)
        centers = km.cluster_centers_
        for c in range(min(n_clusters, len(centers))):
            mask = words == c
            if np.any(mask):
                vlad[c] = (local[mask] - centers[c]).sum(axis=0)
    # intra-normalisation (per cluster)
    vlad /= np.maximum(np.linalg.norm(vlad, axis=1, keepdims=True), 1e-12)
    v = vlad.reshape(-1)
    # signed power-law + global L2
    v = np.sign(v) * np.sqrt(np.abs(v))
    v /= max(float(np.linalg.norm(v)), 1e-12)
    return v.astype(np.float32)
