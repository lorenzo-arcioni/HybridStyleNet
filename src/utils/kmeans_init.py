"""
kmeans_init.py
--------------
Inizializzazione dei cluster via K-Means con criterio del gomito.

Funzioni:
  elbow_kmeans(X, k_max, tau)     — K* + centroidi + assignments
  compute_wcss(X, labels, k)      — varianza intra-cluster (WCSS)
  elbow_k(wcss_list, tau)         — trova K* dalla curva WCSS
  assign_to_clusters(X, centroids)— assegna punti ai centroidi

Usato da IncrementalUpdater (memory/) e dai notebook di adaptation.
Dipende solo da numpy e sklearn (opzionale).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans as _SKLearnKMeans
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# elbow_kmeans  (entry point principale)
# ---------------------------------------------------------------------------

def elbow_kmeans(
    X:     np.ndarray,      # (N, D) float32 — color histograms
    k_max: int   = 15,
    k_min: int   = 1,
    tau:   float = 0.1,
    seed:  int   = 42,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Determina K* con il criterio del gomito ed esegue K-Means finale.

    Criterio del gomito:
      K* = argmin_K { WCSS(K) : [WCSS(K) - WCSS(K+1)] / [WCSS(K-1) - WCSS(K)] > τ }
      (K* è il punto in cui il guadagno marginale scende sotto la soglia τ)

    Parameters
    ----------
    X     : array (N, D) dei color histograms delle coppie di training
    k_max : K massimo da considerare
    k_min : K minimo (default 1)
    tau   : soglia del criterio del gomito
    seed  : random seed per la riproducibilità

    Returns
    -------
    k_star    : int — numero ottimale di cluster
    centroids : (k_star, D) float32 — centroidi K-Means
    assignments: (N,) int — indice cluster per ogni punto
    """
    N = len(X)

    # Con N piccolo si limita k_max
    actual_k_max = min(k_max, N // 2, 15)
    actual_k_min = max(k_min, 1)

    if actual_k_max <= actual_k_min:
        # Degenere: un solo cluster
        centroid    = X.mean(axis=0, keepdims=True)
        assignments = np.zeros(N, dtype=np.int64)
        return 1, centroid, assignments

    # Calcola WCSS per K = k_min..k_max
    wcss_list: List[float] = []
    for k in range(actual_k_min, actual_k_max + 1):
        wcss_list.append(compute_wcss_for_k(X, k, seed=seed))

    # Trova K* con il criterio del gomito
    k_star = elbow_k(
        wcss_list,
        k_offset = actual_k_min,
        tau      = tau,
    )

    # K-Means finale con K*
    centroids, assignments = run_kmeans(X, k=k_star, seed=seed)

    return k_star, centroids, assignments


# ---------------------------------------------------------------------------
# compute_wcss_for_k
# ---------------------------------------------------------------------------

def compute_wcss_for_k(
    X:    np.ndarray,
    k:    int,
    seed: int = 42,
) -> float:
    """
    Calcola la varianza intra-cluster (WCSS) per un dato K.

    WCSS = Σ_k Σ_{x ∈ C_k} ‖x - μ_k‖²
    """
    _, assignments = run_kmeans(X, k=k, seed=seed)
    centroids_k    = np.array([
        X[assignments == c].mean(axis=0) if (assignments == c).any()
        else X.mean(axis=0)
        for c in range(k)
    ])

    wcss = 0.0
    for c in range(k):
        mask   = assignments == c
        if not mask.any():
            continue
        points = X[mask]
        wcss  += ((points - centroids_k[c]) ** 2).sum()

    return float(wcss)


# ---------------------------------------------------------------------------
# elbow_k
# ---------------------------------------------------------------------------

def elbow_k(
    wcss_list: List[float],
    k_offset:  int   = 1,
    tau:       float = 0.1,
) -> int:
    """
    Trova K* dal vettore di valori WCSS.

    Formula:
      ratio(K) = [WCSS(K) - WCSS(K+1)] / [WCSS(K-1) - WCSS(K)]
      K* = primo K con ratio > τ  (guadagno marginale che scende)

    Se nessun K supera τ, restituisce il K con la variazione
    relativa massima (metodo del massimo guadagno normalizzato).

    Parameters
    ----------
    wcss_list : valori WCSS per K = k_offset, k_offset+1, ...
    k_offset  : valore di K per il primo elemento di wcss_list
    tau       : soglia del rapporto
    """
    n = len(wcss_list)
    if n == 1:
        return k_offset

    # Cerca il gomito con il criterio del rapporto
    for i in range(1, n - 1):
        delta_prev = wcss_list[i - 1] - wcss_list[i]
        delta_next = wcss_list[i]     - wcss_list[i + 1]
        if delta_prev < 1e-10:
            continue
        ratio = delta_next / delta_prev
        if ratio < tau:
            return k_offset + i

    # Fallback: massima variazione normalizzata (metodo della differenza seconda)
    if n >= 3:
        second_diffs = [
            abs((wcss_list[i + 1] - wcss_list[i]) - (wcss_list[i] - wcss_list[i - 1]))
            for i in range(1, n - 1)
        ]
        best_i = int(np.argmax(second_diffs)) + 1
        return k_offset + best_i

    # Degenere: restituisce K=2
    return k_offset + 1


# ---------------------------------------------------------------------------
# run_kmeans
# ---------------------------------------------------------------------------

def run_kmeans(
    X:    np.ndarray,
    k:    int,
    seed: int = 42,
    n_init: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Esegue K-Means e restituisce (centroidi, assignments).

    Usa sklearn se disponibile (più robusto), altrimenti
    implementazione numpy minimale.

    Returns
    -------
    centroids   : (k, D) float32
    assignments : (N,) int64
    """
    if _SKLEARN_AVAILABLE:
        km = _SKLearnKMeans(
            n_clusters=k, n_init=n_init,
            random_state=seed, max_iter=300,
        )
        km.fit(X.astype(np.float32))
        return km.cluster_centers_.astype(np.float32), km.labels_.astype(np.int64)

    # Fallback numpy: K-Means++ init + Lloyd iterations
    return _numpy_kmeans(X, k, seed=seed)


# ---------------------------------------------------------------------------
# assign_to_clusters
# ---------------------------------------------------------------------------

def assign_to_clusters(
    X:         np.ndarray,    # (N, D)
    centroids: np.ndarray,    # (K, D)
) -> np.ndarray:
    """
    Assegna ogni punto al centroide più vicino (distanza L2).

    Returns
    -------
    assignments : (N,) int64
    """
    # Distanza quadratica: (N, K)
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]   # (N, K, D)
    dists = (diffs ** 2).sum(axis=-1)                            # (N, K)
    return dists.argmin(axis=-1).astype(np.int64)


# ---------------------------------------------------------------------------
# _numpy_kmeans  (fallback senza sklearn)
# ---------------------------------------------------------------------------

def _numpy_kmeans(
    X:    np.ndarray,
    k:    int,
    seed: int = 42,
    max_iter: int = 300,
    tol:  float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """K-Means minimale in numpy con K-Means++ initialization."""
    rng = np.random.RandomState(seed)
    N, D = X.shape

    # K-Means++ init
    centroids = [X[rng.randint(N)]]
    for _ in range(1, k):
        dists = np.array([
            min(np.sum((x - c) ** 2) for c in centroids)
            for x in X
        ])
        probs = dists / (dists.sum() + 1e-10)
        cumprobs = probs.cumsum()
        r = rng.rand()
        idx = np.searchsorted(cumprobs, r)
        centroids.append(X[min(idx, N - 1)])

    centroids = np.stack(centroids, axis=0).astype(np.float32)

    # Lloyd iterations
    for _ in range(max_iter):
        assignments = assign_to_clusters(X, centroids)

        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = X[mask].mean(axis=0)
            else:
                new_centroids[c] = centroids[c]

        shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=-1).max())
        centroids = new_centroids
        if shift < tol:
            break

    return centroids, assign_to_clusters(X, centroids)
