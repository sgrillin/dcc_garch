from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

try:
    from scipy.optimize import minimize
    from scipy.special import gammaln
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False
    def gammaln(x):
        return np.log(np.abs(np.math.gamma(x)))

from .univariate import UGARCH
from .utils import ensure_2d, ridge_pd, half_life

@dataclass
class DCCResult:
    a: float
    b: float
    g: Optional[float]
    nu: Optional[float]
    Q_t: np.ndarray
    R_t: np.ndarray
    H_t: np.ndarray
    D_t: np.ndarray
    z_t: np.ndarray
    mu: np.ndarray
    ugarch_params: List[Dict[str, float]]
    success: bool
    message: str
    dist: str
    persistence: float
    corr_half_life: float

    def to_frame(self):
        return dcc_results_df(self)

    def summary(self) -> str:
        return dcc_summary(self)

class DCC:
    """DCC(1,1)-GARCH with options: Gaussian/Student-t and ADCC asymmetry."""
    def __init__(self,
                 mean: Optional[str] = "constant",
                 dist: str = "gaussian",
                 asym: bool = False,
                 ridge: float = 1e-10,
                 init: str = "sample",
                 bounds_abg: Tuple[Tuple[float,float],Tuple[float,float],Tuple[float,float]] = ((1e-6,0.98),(1e-6,0.98),(0.0,0.5)),
                 bounds_nu: Tuple[float,float] = (2.05, 200.0),
                 pen: float = 1e6,
                 return_result: bool = True):
        assert dist in {"gaussian","student"}
        self.mean = mean; self.dist = dist; self.asym = asym
        self.ridge = ridge; self.init = init
        self.bounds_abg = bounds_abg; self.bounds_nu = bounds_nu
        self.pen = pen; self.return_result = return_result
        self.result_: Optional[DCCResult] = None

    def get_params(self) -> dict:
        return {"mean": self.mean, "dist": self.dist, "asym": self.asym, "ridge": self.ridge,
                "init": self.init, "bounds_abg": self.bounds_abg, "bounds_nu": self.bounds_nu,
                "pen": self.pen, "return_result": self.return_result}

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def _dcc_filter(self, z: np.ndarray, a: float, b: float, g: float, Qbar: np.ndarray):
        T, N = z.shape
        Q_t = np.zeros((T, N, N), dtype=float)
        R_t = np.zeros_like(Q_t)
        Q_prev = Qbar.copy()
        for t in range(T):
            outer = np.outer(z[t-1], z[t-1]) if t > 0 else Qbar
            if self.asym:
                nprev = np.minimum(z[t-1], 0.0) if t > 0 else np.zeros(N)
                asym_term = g * np.outer(nprev, nprev)
            else:
                asym_term = 0.0
            Q = (1 - a - b) * Qbar + a * outer + b * Q_prev + asym_term
            Q[np.diag_indices_from(Q)] += self.ridge
            Q_t[t] = Q
            d = np.sqrt(np.clip(np.diag(Q), 1e-12, None))
            invd = 1.0 / np.clip(d, 1e-12, None)
            R = (invd[:, None] * Q) * invd[None, :]
            R[np.diag_indices_from(R)] = 1.0
            R_t[t] = R
            Q_prev = Q
        return Q_t, R_t

    def _neg_ll_corr_gaussian(self, z, a, b, g, Qbar):
        penalty = 0.0
        if a < 0 or b < 0 or (self.asym and g < 0):
            return 1e12
        if a + b + (0.5*(g if self.asym else 0.0)) >= 0.999:
            penalty += self.pen * (a + b + (0.5*(g if self.asym else 0.0)) - 0.999 + 1e-12) ** 2
        T = z.shape[0]
        Q_t, R_t = self._dcc_filter(z, a, b, g if self.asym else 0.0, Qbar)
        val = 0.0
        for t in range(T):
            R = R_t[t]
            sign, logdet = np.linalg.slogdet(R)
            if sign <= 0:
                return 1e12
            invR = np.linalg.inv(R)
            quad = z[t] @ invR @ z[t]
            val += 0.5 * (logdet + quad - z[t] @ z[t])
        return float(val + penalty)

    def _neg_ll_corr_student(self, z, a, b, g, Qbar, nu):
        if nu <= 2.01:
            return 1e12
        penalty = 0.0
        if a < 0 or b < 0 or (self.asym and g < 0):
            return 1e12
        if a + b + (0.5*(g if self.asym else 0.0)) >= 0.999:
            penalty += self.pen * (a + b + (0.5*(g if self.asym else 0.0)) - 0.999 + 1e-12) ** 2
        T, N = z.shape
        Q_t, R_t = self._dcc_filter(z, a, b, g if self.asym else 0.0, Qbar)
        const = (gammaln((nu + N) / 2.0) - gammaln(nu / 2.0) - (N/2.0)*np.log(nu * np.pi))
        val = 0.0
        for t in range(T):
            R = R_t[t]
            sign, logdet = np.linalg.slogdet(R)
            if sign <= 0:
                return 1e12
            invR = np.linalg.inv(R)
            quad = z[t] @ invR @ z[t]
            val += 0.5 * logdet + 0.5 * (nu + N) * np.log(1.0 + quad / nu) - const
        return float(val + penalty)

    def fit(self, returns: np.ndarray):
        X = ensure_2d(returns)
        T, N = X.shape

        # ----- Univariate options (optional overrides) -----
        u_mean       = getattr(self, "uni_mean", self.mean)                 # 'constant'|'arma'|'arima'
        u_mean_order = getattr(self, "uni_mean_order", (0, 0, 0))           # (p,d,q) for arma/arima
        u_vol        = getattr(self, "uni_vol", "garch")                    # 'garch'|'gjr'|'egarch'
        u_dist       = getattr(self, "uni_dist", "gaussian")                # 'gaussian'|'student'
        u_kwargs     = getattr(self, "uni_kwargs", {})                      # extra kwargs for UGARCH

        # ----- Preallocate -----
        ugarch_params: List[Dict[str, float]] = []
        mu    = np.empty(N, dtype=float)
        eps   = np.empty_like(X, dtype=float)
        sigma = np.empty_like(X, dtype=float)

        # ----- Univariate fits (per series) -----
        for i in range(N):
            model = UGARCH(mean=u_mean, mean_order=u_mean_order, vol=u_vol, dist=u_dist, **u_kwargs)
            res = model.fit(X[:, i])

            if not getattr(res, "success", True):
                raise RuntimeError(f"Univariate fit failed for series {i}: {getattr(res, 'message', 'no message')}")

            ugarch_params.append(res.params)
            mu[i]       = res.mu
            eps[:, i]   = res.eps
            sigma[:, i] = np.sqrt(res.sigma2)  # conditional std dev

        # ----- Standardized residuals -----
        z = eps / np.clip(sigma, 1e-12, None)

        # ----- Long-run target Qbar (with optional shrinkage) -----
        if self.init == "identity":
            Qbar = np.eye(N)
        else:
            # sample correlation of z
            Qbar = np.cov(z.T)
            d = np.sqrt(np.diag(Qbar)); invd = 1.0 / np.clip(d, 1e-12, None)
            Qbar = (invd[:, None] * Qbar) * invd[None, :]

            # optional ridge/shrinkage toward identity if attribute present
            lam = float(getattr(self, "qbar_shrink", 0.0))
            if 0.0 < lam < 1.0:
                Qbar = (1.0 - lam) * Qbar + lam * np.eye(N)

        # ----- Starting values -----
        a0, b0 = 0.02, 0.95
        g0 = 0.03 if self.asym else 0.0

        # ----- Optimize correlation params -----
        if self.dist == "gaussian":
            def obj(x):
                a, b = x[0], x[1]
                g = x[2] if self.asym else 0.0
                return self._neg_ll_corr_gaussian(z, a, b, g, Qbar)

            if self.asym:
                bounds = (self.bounds_abg[0], self.bounds_abg[1], self.bounds_abg[2])
                x0 = np.array([a0, b0, g0], dtype=float)
            else:
                bounds = (self.bounds_abg[0], self.bounds_abg[1])
                x0 = np.array([a0, b0], dtype=float)

            if _HAVE_SCIPY:
                res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
                xhat = res.x; success, msg = bool(res.success), str(res.message)
            else:
                grid = np.linspace(0.01, 0.98, 20)
                best = (np.inf, x0.copy())
                for a in grid:
                    for b in grid:
                        if self.asym:
                            for g in np.linspace(0.0, 0.4, 6):
                                val = obj(np.array([a, b, g], dtype=float))
                                if val < best[0]: best = (val, np.array([a, b, g], float))
                        else:
                            val = obj(np.array([a, b], dtype=float))
                            if val < best[0]: best = (val, np.array([a, b], float))
                xhat = best[1]; success, msg = True, "SciPy not available; used coarse grid search."

            a, b = float(xhat[0]), float(xhat[1])
            g = float(xhat[2]) if self.asym else None
            nu = None

        else:  # Student-t DCC
            def obj(x):
                a, b = x[0], x[1]
                if self.asym:
                    g = x[2]; nu = x[3]
                else:
                    g = 0.0; nu = x[2]
                if not (self.bounds_nu[0] < nu < self.bounds_nu[1]):
                    return 1e12
                return self._neg_ll_corr_student(z, a, b, g, Qbar, nu)

            if self.asym:
                bounds = (self.bounds_abg[0], self.bounds_abg[1], self.bounds_abg[2], self.bounds_nu)
                x0 = np.array([a0, b0, g0, 8.0], dtype=float)
            else:
                bounds = (self.bounds_abg[0], self.bounds_abg[1], self.bounds_nu)
                x0 = np.array([a0, b0, 8.0], dtype=float)

            if _HAVE_SCIPY:
                res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
                xhat = res.x; success, msg = bool(res.success), str(res.message)
            else:
                grid = np.linspace(0.01, 0.98, 15); nu_grid = np.linspace(4.0, 30.0, 10)
                best = (np.inf, x0.copy())
                for a in grid:
                    for b in grid:
                        for nu_val in nu_grid:
                            if self.asym:
                                for g_val in np.linspace(0.0, 0.4, 6):
                                    x = np.array([a, b, g_val, nu_val], float); val = obj(x)
                                    if val < best[0]: best = (val, x)
                            else:
                                x = np.array([a, b, nu_val], float); val = obj(x)
                                if val < best[0]: best = (val, x)
                xhat = best[1]; success, msg = True, "SciPy not available; used coarse grid search."

            a, b = float(xhat[0]), float(xhat[1])
            g  = float(xhat[2]) if self.asym else None
            nu = float(xhat[3]) if self.asym else float(xhat[2])

        # ----- Filter correlation path with estimates -----
        Q_t, R_t = self._dcc_filter(z, a, b, (g if g is not None else 0.0), Qbar)

        # ----- Build D_t from exact univariate sigmas and H_t = D_t R_t D_t -----
        D_t = np.zeros((T, N, N), dtype=float)
        for t in range(T):
            D_t[t] = np.diag(np.clip(sigma[t], 1e-12, None))
        H_t = np.einsum("tij,tjk,tkm->tim", D_t, R_t, D_t)  # (T,N,N)

        # ----- Persistence & half-life -----
        persistence = a + b + (0.5 * (g if g is not None else 0.0))
        hl = half_life(persistence) if persistence < 1 else np.inf

        # ----- Pack results -----
        result = DCCResult(
            a=a, b=b, g=g, nu=nu,
            Q_t=Q_t, R_t=R_t, H_t=H_t, D_t=D_t,
            z_t=z, mu=mu, ugarch_params=ugarch_params,
            success=success, message=msg, dist=self.dist,
            persistence=persistence, corr_half_life=hl
        )
        self.result_ = result
        return result if self.return_result else self

    def forecast(self, steps: int = 1) -> Dict[str, np.ndarray]:
        if self.result_ is None:
            raise RuntimeError("Call fit() first.")
        a, b = self.result_.a, self.result_.b
        g = self.result_.g if (self.result_.g is not None) else 0.0
        Q_t = self.result_.Q_t; R_t = self.result_.R_t; D_t = self.result_.D_t; z_t = self.result_.z_t
        T, N = z_t.shape
        Qbar = np.cov(z_t.T); d = np.sqrt(np.diag(Qbar)); invd = 1.0 / np.clip(d, 1e-12, None)
        Qbar = (invd[:, None] * Qbar) * invd[None, :]

        Q_last = Q_t[-1].copy(); D_last = D_t[-1].copy(); z_last = z_t[-1].copy()
        Q_fore, R_fore, H_fore = [], [], []
        for h in range(steps):
            outer = np.outer(z_last, z_last) if h == 0 else R_fore[-1]
            if self.asym:
                nprev = np.minimum(z_last, 0.0) if h == 0 else np.zeros_like(z_last)
                asym_term = g * np.outer(nprev, nprev)
            else:
                asym_term = 0.0
            Q_next = (1 - a - b) * Qbar + a * outer + b * Q_last + asym_term
            Q_next[np.diag_indices_from(Q_next)] += 1e-10
            d = np.sqrt(np.diag(Q_next)); invd = 1.0 / np.clip(d, 1e-12, None)
            R_next = (invd[:, None] * Q_next) * invd[None, :]
            H_next = D_last @ R_next @ D_last
            Q_fore.append(Q_next)
            R_fore.append(R_next)
            H_fore.append(H_next)
            Q_last = Q_next
        return {"Q": np.stack(Q_fore, axis=0), "R": np.stack(R_fore, axis=0), "H": np.stack(H_fore, axis=0)}

class RollingDCC:
    def __init__(self, window: Optional[int] = 500, step: int = 1, expanding: bool = False, min_samples: int = 250, **dcc_kwargs):
        self.window = window; self.step = step; self.expanding = expanding; self.min_samples = min_samples
        self.dcc_kwargs = dcc_kwargs
        self.results_: List[DCCResult] = []

    def fit(self, returns: np.ndarray) -> "RollingDCC":
        X = ensure_2d(returns)
        T = X.shape[0]
        starts = []
        if self.expanding:
            s = self.min_samples
            while s <= T:
                starts.append((0, s)); s += self.step
        else:
            if self.window is None:
                raise ValueError("Provide window for rolling, or set expanding=True.")
            s = 0
            while s + self.window <= T:
                starts.append((s, s + self.window)); s += self.step
        self.results_.clear()
        for (i, j) in starts:
            dcc = DCC(**self.dcc_kwargs)
            res = dcc.fit(X[i:j, :])
            self.results_.append(res)
        return self

    def last_covariances(self) -> np.ndarray:
        if not self.results_:
            raise RuntimeError("Call fit() first.")
        mats = [r.H_t[-1] for r in self.results_]
        return np.stack(mats, axis=0)

    def last_correlations(self) -> np.ndarray:
        if not self.results_:
            raise RuntimeError("Call fit() first.")
        mats = [r.R_t[-1] for r in self.results_]
        return np.stack(mats, axis=0)


def dcc_results_df(res) -> pd.DataFrame:
    """
    Return DCC parameter estimates (a, b, [g], [nu]) in a tidy DataFrame.
    (SEs omitted unless you implement inference for DCC params.)
    """
    rows = [("a", float(res.a)), ("b", float(res.b))]
    if res.g is not None:
        rows.append(("g", float(res.g)))
    if res.nu is not None:
        rows.append(("nu", float(res.nu)))
    df = pd.DataFrame(rows, columns=["param", "estimate"]).set_index("param")
    return df

def dcc_summary(res) -> str:
    df = dcc_results_df(res)
    lines = []
    lines.append("DCC Results")
    lines.append("-" * 60)
    lines.append(f"dist: {res.dist} | asym: {res.g is not None}")
    lines.append(f"converged: {res.success} | message: {res.message}")
    lines.append(f"persistence: {res.persistence:.6g} | corr half-life: {res.corr_half_life:.6g}")
    lines.append("")
    lines.append(df.to_string(float_format=lambda x: f"{x:.6g}"))
    return "\n".join(lines)
# %%
