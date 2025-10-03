import numpy as np

def ensure_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

def ridge_pd(mat, eps=1e-10):
    m = np.array(mat, dtype=float, copy=True)
    m[np.diag_indices_from(m)] += eps
    return m

def safe_invert(mat, ridge=1e-10):
    m = np.array(mat, dtype=float, copy=True)
    m[np.diag_indices_from(m)] += ridge
    return np.linalg.inv(m)

def half_life(persistence):
    if persistence <= 0:
        return np.inf
    return np.log(0.5) / np.log(persistence)

def block_diag_from_std(std_vec):
    d = np.asarray(std_vec, dtype=float).ravel()
    return np.diag(d)

def as_float(x):
    return float(x)
