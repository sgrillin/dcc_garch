from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
try:
    from scipy.optimize import minimize
    from scipy.special import gammaln
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False
    def gammaln(x):
        return np.log(np.abs(np.math.gamma(x)))

@dataclass
class UGARCHResult:
    params: Dict[str, float]
    sigma2: np.ndarray
    eps: np.ndarray
    mu: float
    success: bool
    message: str
    dist: str

class UGARCH:
    """Univariate GARCH(1,1) with Gaussian or Student-t QMLE."""
    def __init__(self, mean: Optional[str] = "constant", dist: str = "gaussian",
                 bounds: Optional[dict] = None, pen: float = 1e6):
        assert dist in {"gaussian", "student"}
        self.mean = mean
        self.dist = dist
        self.bounds = bounds
        self.pen = pen
        self.result_: Optional[UGARCHResult] = None

    @staticmethod
    def _garch_filter(eps: np.ndarray, omega: float, alpha: float, beta: float, sigma2_0: Optional[float] = None) -> np.ndarray:
        T = len(eps)
        sigma2 = np.empty(T, dtype=float)
        if sigma2_0 is None:
            if alpha + beta < 0.999:
                sigma2_0 = omega / max(1e-12, (1.0 - alpha - beta))
            else:
                sigma2_0 = float(np.var(eps, ddof=1))
        s_prev = sigma2_0
        for t in range(T):
            s_t = omega + alpha * (eps[t-1] ** 2 if t > 0 else 0.0) + beta * s_prev
            sigma2[t] = max(s_t, 1e-12)
            s_prev = sigma2[t]
        return sigma2

    def _neg_ll_gaussian(self, x: np.ndarray, eps: np.ndarray) -> float:
        omega, alpha, beta = x
        if omega <= 0 or alpha < 0 or beta < 0:
            return 1e12
        penalty = 0.0
        if alpha + beta >= 0.999:
            penalty += self.pen * (alpha + beta - 0.999 + 1e-12) ** 2
        sigma2 = self._garch_filter(eps, omega, alpha, beta)
        ll = 0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (eps ** 2) / sigma2)
        return float(np.sum(ll) + penalty)

    def _neg_ll_student(self, x: np.ndarray, eps: np.ndarray) -> float:
        omega, alpha, beta, nu = x
        if omega <= 0 or alpha < 0 or beta < 0 or nu <= 2.01:
            return 1e12
        penalty = 0.0
        if alpha + beta >= 0.999:
            penalty += self.pen * (alpha + beta - 0.999 + 1e-12) ** 2
        sigma2 = self._garch_filter(eps, omega, alpha, beta)
        const = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log((nu - 2.0) * np.pi))
        term = 0.5 * np.log(sigma2) + ((nu + 1.0)/2.0) * np.log(1.0 + (eps**2) / ((nu - 2.0) * sigma2))
        nll = np.sum(term) - len(eps) * const
        return float(nll + penalty)

    def get_params(self) -> dict:
        return {"mean": self.mean, "dist": self.dist, "bounds": self.bounds, "pen": self.pen}

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def fit(self, r: np.ndarray) -> UGARCHResult:
        r = np.asarray(r, dtype=float).ravel()
        mu = float(np.mean(r)) if self.mean == "constant" else 0.0
        eps = r - mu
        var = np.var(eps, ddof=1) + 1e-8
        omega0, alpha0, beta0 = 0.01 * var, 0.05, 0.90

        if self.dist == "gaussian":
            x0 = np.array([omega0, alpha0, beta0], dtype=float)
            b = self.bounds or {"omega": (1e-12, 10 * var), "alpha": (0.0, 0.999), "beta": (0.0, 0.999)}
            bounds = (b["omega"], b["alpha"], b["beta"])
            if _HAVE_SCIPY:
                res = minimize(self._neg_ll_gaussian, x0, args=(eps,), method="L-BFGS-B", bounds=bounds)
                omega, alpha, beta = res.x; success = bool(res.success); message = str(res.message)
            else:
                success, message = True, "Scipy not available; used coarse grid search."
                grid = np.linspace(0.01, 0.98, 25)
                best = (np.inf, x0)
                for a in grid:
                    for bb in grid:
                        if a + bb >= 0.999: 
                            continue
                        omega = 0.01 * var * (1 - a - bb)
                        x = np.array([omega, a, bb])
                        val = self._neg_ll_gaussian(x, eps)
                        if val < best[0]:
                            best = (val, x)
                omega, alpha, beta = best[1]
            sigma2 = self._garch_filter(eps, omega, alpha, beta)
            params = {"omega": float(omega), "alpha": float(alpha), "beta": float(beta)}
        else:
            x0 = np.array([omega0, alpha0, beta0, 8.0], dtype=float)
            b = self.bounds or {"omega": (1e-12, 10 * var), "alpha": (0.0, 0.999), "beta": (0.0, 0.999), "nu": (2.05, 200.0)}
            bounds = (b["omega"], b["alpha"], b["beta"], b["nu"])
            if _HAVE_SCIPY:
                res = minimize(self._neg_ll_student, x0, args=(eps,), method="L-BFGS-B", bounds=bounds)
                omega, alpha, beta, nu = res.x; success = bool(res.success); message = str(res.message)
            else:
                success, message = True, "Scipy not available; used coarse grid grid for a,b and fixed nu=8."
                grid = np.linspace(0.01, 0.98, 25)
                best = (np.inf, x0)
                for a in grid:
                    for bb in grid:
                        if a + bb >= 0.999: 
                            continue
                        omega = 0.01 * var * (1 - a - bb)
                        x = np.array([omega, a, bb, 8.0])
                        val = self._neg_ll_student(x, eps)
                        if val < best[0]:
                            best = (val, x)
                omega, alpha, beta, nu = best[1]
            sigma2 = self._garch_filter(eps, omega, alpha, beta)
            params = {"omega": float(omega), "alpha": float(alpha), "beta": float(beta), "nu": float(nu)}

        out = UGARCHResult(params=params, sigma2=sigma2, eps=eps, mu=mu,
                           success=success, message=message, dist=self.dist)
        self.result_ = out
        return out
