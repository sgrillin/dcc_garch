from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

# --- Optional deps ---
try:
    from scipy.optimize import minimize
    from scipy.special import gammaln
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False
    def gammaln(x):
        return np.log(np.abs(np.math.gamma(x)))

try:
    from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
    _HAVE_STATSMODELS = True
except Exception:  # pragma: no cover
    _HAVE_STATSMODELS = False


# ---------- Results container ----------
@dataclass
class UGARCHResult:
    params: Dict[str, float]
    sigma2: np.ndarray
    eps: np.ndarray
    mu: float  # if mean model is dynamic, this is 0; residuals already account for it
    success: bool
    message: str
    dist: str
    vol: str = "garch"
    mean: str = "constant"
    mean_order: Tuple[int, int, int] = (0, 0, 0)
    # Optional inference
    vcov: Optional[np.ndarray] = None
    vcov_robust: Optional[np.ndarray] = None
    se: Optional[np.ndarray] = None
    se_robust: Optional[np.ndarray] = None
    param_names: Optional[List[str]] = None

    def to_frame(self):
        return ugarch_results_df(self)

    def summary(self) -> str:
        return ugarch_summary(self)


# ---------- Model ----------
class UGARCH:
    """
    Univariate volatility models with QMLE:
      - Mean:     constant | ARMA(p,q) | ARIMA(p,d,q)
      - Vol:      GARCH(1,1) | GJR-GARCH(1,1) | EGARCH(1,1)
      - Dist:     Gaussian | Student-t (EGARCH currently Gaussian only)
    """

    def __init__(
        self,
        mean: Optional[str] = "constant",
        mean_order: Tuple[int, int, int] = (0, 0, 0),  # (p,d,q). For ARMA use d=0
        vol: str = "garch",  # "garch" | "gjr" | "egarch"
        dist: str = "gaussian",  # "gaussian" | "student"
        bounds: Optional[dict] = None,
        pen: float = 1e6,
        compute_se: bool = False,
        robust_se: bool = False,
        fd_step: float = 1e-5,
    ):
        assert dist in {"gaussian", "student"}
        assert vol in {"garch", "gjr", "egarch"}
        assert mean in {"constant", "arma", "arima"}
        self.mean = mean
        self.mean_order = tuple(mean_order)
        self.vol = vol
        self.dist = dist
        self.bounds = bounds
        self.pen = pen
        self.compute_se = compute_se
        self.robust_se = robust_se
        self.fd_step = fd_step
        self.result_: Optional[UGARCHResult] = None

    # ----- Mean step (returns residuals eps) -----
    def _mean_filter(self, r: np.ndarray) -> Tuple[np.ndarray, float]:
        r = np.asarray(r, float).ravel()
        if self.mean == "constant":
            mu = float(np.mean(r))
            eps = r - mu
            return eps, mu

        if not _HAVE_STATSMODELS:
            raise RuntimeError("statsmodels is required for mean='arma' or 'arima'.")

        p, d, q = self.mean_order
        # statsmodels handles both ARMA (d=0) and ARIMA (d>0).
        # We use in-sample residuals from the fitted ARIMA.
        model = SM_ARIMA(r, order=(int(p), int(d), int(q)))
        fit = model.fit(method_kwargs={"warn_convergence": False})
        eps = np.asarray(fit.resid, float)
        # We store mu=0 because eps already accounts for the dynamic mean.
        return eps, 0.0

    # ----- Volatility recursions -----
    @staticmethod
    def _garch_filter(eps: np.ndarray, omega: float, alpha: float, beta: float,
                      sigma2_0: Optional[float] = None) -> np.ndarray:
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

    @staticmethod
    def _gjr_filter(eps: np.ndarray, omega: float, alpha: float, beta: float, gamma: float,
                    sigma2_0: Optional[float] = None) -> np.ndarray:
        # σ_t^2 = ω + α ε_{t-1}^2 + γ I(ε_{t-1}<0) ε_{t-1}^2 + β σ_{t-1}^2
        T = len(eps)
        sigma2 = np.empty(T, dtype=float)
        if sigma2_0 is None:
            # use unconditional variance approximation with γ/2 contribution
            persistence = alpha + beta + 0.5 * max(0.0, gamma)
            if persistence < 0.999:
                sigma2_0 = omega / max(1e-12, (1.0 - persistence))
            else:
                sigma2_0 = float(np.var(eps, ddof=1))
        s_prev = sigma2_0
        for t in range(T):
            e2m = 0.0
            if t > 0:
                neg = 1.0 if eps[t-1] < 0 else 0.0
                e2m = (alpha + gamma * neg) * (eps[t-1] ** 2)
            s_t = omega + e2m + beta * s_prev
            sigma2[t] = max(s_t, 1e-12)
            s_prev = sigma2[t]
        return sigma2

    @staticmethod
    def _egarch_filter(eps: np.ndarray, omega: float, alpha: float, gamma: float, beta: float,
                       log_sigma2_0: Optional[float] = None, dist: str = "gaussian", nu: Optional[float] = None) -> np.ndarray:
        # log σ_t^2 = ω + β log σ_{t-1}^2 + α (|z_{t-1}| - c) + γ z_{t-1}
        # with z_{t-1} = ε_{t-1} / σ_{t-1}; c = E|Z| (Gaussian: sqrt(2/pi))
        if dist != "gaussian":
            raise NotImplementedError("EGARCH implemented with Gaussian innovations only.")
        T = len(eps)
        log_sigma2 = np.empty(T, dtype=float)
        if log_sigma2_0 is None:
            log_sigma2_0 = float(np.log(np.var(eps, ddof=1) + 1e-12))
        ls_prev = log_sigma2_0
        c = np.sqrt(2.0 / np.pi)  # E|Z| under Gaussian
        for t in range(T):
            if t == 0:
                zprev = 0.0
            else:
                sig_prev = float(np.exp(0.5 * ls_prev))
                zprev = eps[t-1] / max(sig_prev, 1e-12)
            ls_t = omega + beta * ls_prev + alpha * (abs(zprev) - c) + gamma * zprev
            log_sigma2[t] = ls_t
            ls_prev = ls_t
        sigma2 = np.exp(log_sigma2)
        sigma2[sigma2 < 1e-12] = 1e-12
        return sigma2

    # ----- Negative log-likelihoods (return sum over t) -----
    def _nll_gaussian(self, eps: np.ndarray, sigma2: np.ndarray) -> float:
        return 0.5 * float(np.sum(np.log(2 * np.pi) + np.log(sigma2) + (eps ** 2) / sigma2))

    def _nll_student(self, eps: np.ndarray, sigma2: np.ndarray, nu: float) -> float:
        if nu <= 2.01:
            return 1e12
        const = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log((nu - 2.0) * np.pi))
        term = 0.5 * np.log(sigma2) + ((nu + 1.0) / 2.0) * np.log(1.0 + (eps ** 2) / ((nu - 2.0) * sigma2))
        nll = np.sum(term) - len(eps) * const
        return float(nll)

    # ----- Model-specific wrappers with constraints / penalties -----
    def _neg_ll(self, x: np.ndarray, eps: np.ndarray) -> float:
        # Route based on vol + dist. x packs parameters in a consistent order per model.
        v = self.vol
        d = self.dist
        pen = 0.0

        if v == "garch":
            if d == "gaussian":
                omega, alpha, beta = x
            else:  # student
                omega, alpha, beta, nu = x
            # constraints
            if omega <= 0 or alpha < 0 or beta < 0:
                return 1e12
            if alpha + beta >= 0.999:
                pen += self.pen * (alpha + beta - 0.999 + 1e-12) ** 2
            sigma2 = self._garch_filter(eps, omega, alpha, beta)
            return (self._nll_gaussian(eps, sigma2) if d == "gaussian" else self._nll_student(eps, sigma2, nu)) + pen

        elif v == "gjr":
            if d == "gaussian":
                omega, alpha, beta, gamma = x
            else:
                omega, alpha, beta, gamma, nu = x
            if omega <= 0 or alpha < 0 or beta < 0 or gamma < 0:
                return 1e12
            persistence = alpha + beta + 0.5 * gamma
            if persistence >= 0.999:
                pen += self.pen * (persistence - 0.999 + 1e-12) ** 2
            sigma2 = self._gjr_filter(eps, omega, alpha, beta, gamma)
            return (self._nll_gaussian(eps, sigma2) if d == "gaussian" else self._nll_student(eps, sigma2, nu)) + pen

        else:  # EGARCH
            # We use param order: [omega, alpha, gamma, beta]  (beta is AR(1) on log σ²)
            if d != "gaussian":
                return 1e12  # keep it simple/stable
            omega, alpha, gamma, beta = x
            # Typical stability: |beta|<1; alpha, gamma free-ish; omega free
            if not (-0.999 < beta < 0.999):
                return 1e12
            sigma2 = self._egarch_filter(eps, omega, alpha, gamma, beta, dist=d)
            return self._nll_gaussian(eps, sigma2)

    # ----- Public helpers -----
    def get_params(self) -> dict:
        return {
            "mean": self.mean, "mean_order": self.mean_order,
            "vol": self.vol, "dist": self.dist,
            "bounds": self.bounds, "pen": self.pen,
            "compute_se": self.compute_se, "robust_se": self.robust_se,
            "fd_step": self.fd_step,
        }

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    # ----- Fit -----
    def fit(self, r: np.ndarray) -> UGARCHResult:
        # Mean step -> residuals for variance step
        eps, mu = self._mean_filter(r)
        var = np.var(eps, ddof=1) + 1e-8

        # Initial guesses and parameterization
        if self.vol == "garch":
            # [omega, alpha, beta] (+ nu)
            omega0, alpha0, beta0 = 0.01 * var, 0.05, 0.90
            x0 = [omega0, alpha0, beta0]
            names = ["omega", "alpha", "beta"]
            if self.dist == "student":
                x0 += [8.0]; names += ["nu"]
            b = self.bounds or (
                (1e-12, 10 * var),  # omega
                (0.0, 0.999),       # alpha
                (0.0, 0.999),       # beta
            )
            if self.dist == "student":
                bounds = b + ((2.05, 200.0),)
            else:
                bounds = b

        elif self.vol == "gjr":
            # [omega, alpha, beta, gamma] (+ nu)
            omega0, alpha0, beta0, gamma0 = 0.01 * var, 0.05, 0.90, 0.05
            x0 = [omega0, alpha0, beta0, gamma0]
            names = ["omega", "alpha", "beta", "gamma"]
            if self.dist == "student":
                x0 += [8.0]; names += ["nu"]
            b = self.bounds or (
                (1e-12, 10 * var),  # omega
                (0.0, 0.999),       # alpha
                (0.0, 0.999),       # beta
                (0.0, 1.0),         # gamma >= 0
            )
            if self.dist == "student":
                bounds = b + ((2.05, 200.0),)
            else:
                bounds = b

        else:  # EGARCH
            # [omega, alpha, gamma, beta]  (Gaussian only here)
            # wide bounds for omega/alpha/gamma, stable beta in (-0.999,0.999)
            x0 = [np.log(var + 1e-8), 0.10, -0.05, 0.90]
            names = ["omega", "alpha", "gamma", "beta"]
            bounds = self.bounds or ((-20.0, 20.0), (-5.0, 5.0), (-5.0, 5.0), (-0.999, 0.999))

        x0 = np.array(x0, float)

        # Optimize
        if _HAVE_SCIPY:
            res = minimize(self._neg_ll, x0, args=(eps,), method="L-BFGS-B", bounds=bounds)
            xhat = np.array(res.x, float)
            success = bool(res.success)
            message = str(res.message)
        else:
            # Fallback coarse grid (very crude; relies on persistence structure)
            success, message = True, "SciPy not available; used coarse grid search."
            def grid(vals):
                best = (np.inf, None)
                for a in vals:
                    for b in vals:
                        if self.vol == "garch":
                            if a + b >= 0.999: continue
                            omega = 0.01 * var * (1 - a - b)
                            xx = np.array([omega, a, b] + ([8.0] if self.dist == "student" else []))
                        elif self.vol == "gjr":
                            for g in np.linspace(0.0, 0.6, 7):
                                if a + b + 0.5 * g >= 0.999: continue
                                omega = 0.01 * var * (1 - a - b - 0.5*g)
                                xx = np.array([omega, a, b, g] + ([8.0] if self.dist == "student" else []))
                                val = self._neg_ll(xx, eps) 
                                if val < best[0]: best = (val, xx)
                            continue
                        else:  # egarch
                            for g in np.linspace(-0.3, 0.3, 7):
                                beta = b
                                xx = np.array([np.log(var+1e-8), a, g, beta])
                                val = self._neg_ll(xx, eps)
                                if val < best[0]: best = (val, xx)
                            continue
                        val = self._neg_ll(xx, eps)
                        if val < best[0]: best = (val, xx)
                return best[1]
            vals = np.linspace(0.02, 0.96, 16)
            xhat = grid(vals)

        # Build outputs
        if self.vol == "garch":
            if self.dist == "gaussian":
                omega, alpha, beta = xhat
                nu = None
                sigma2 = self._garch_filter(eps, omega, alpha, beta)
                params = {"omega": float(omega), "alpha": float(alpha), "beta": float(beta)}
            else:
                omega, alpha, beta, nu = xhat
                sigma2 = self._garch_filter(eps, omega, alpha, beta)
                params = {"omega": float(omega), "alpha": float(alpha), "beta": float(beta), "nu": float(nu)}

        elif self.vol == "gjr":
            if self.dist == "gaussian":
                omega, alpha, beta, gamma = xhat
                nu = None
                sigma2 = self._gjr_filter(eps, omega, alpha, beta, gamma)
                params = {"omega": float(omega), "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma)}
            else:
                omega, alpha, beta, gamma, nu = xhat
                sigma2 = self._gjr_filter(eps, omega, alpha, beta, gamma)
                params = {"omega": float(omega), "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma), "nu": float(nu)}

        else:  # EGARCH
            omega, alpha, gamma, beta = xhat; nu = None
            sigma2 = self._egarch_filter(eps, omega, alpha, gamma, beta, dist=self.dist)
            params = {"omega": float(omega), "alpha": float(alpha), "gamma": float(gamma), "beta": float(beta)}

        out = UGARCHResult(
            params=params, sigma2=sigma2, eps=eps, mu=mu,
            success=success, message=message, dist=self.dist,
            vol=self.vol, mean=self.mean, mean_order=self.mean_order,
            param_names=names
        )

        # Optional inference
        if (self.compute_se or self.robust_se) and _HAVE_SCIPY:
            vcov, vcov_rob, se, se_rob = self._compute_inference(eps, xhat)
            out.vcov, out.vcov_robust, out.se, out.se_robust = vcov, vcov_rob, se, se_rob

        self.result_ = out
        return out

    # ----- Inference: numerical Hessian + OPG (robust) -----
    def _contribs(self, x: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """
        Per-observation negative log-likelihood contributions at params x.
        Re-runs the filter once; used for robust SE via OPG (sandwich).
        """
        v, d = self.vol, self.dist
        if v == "garch":
            if d == "gaussian":
                omega, alpha, beta = x
                sigma2 = self._garch_filter(eps, omega, alpha, beta)
                ll = 0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (eps ** 2) / sigma2)
                return ll
            else:
                omega, alpha, beta, nu = x
                sigma2 = self._garch_filter(eps, omega, alpha, beta)
                const = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log((nu - 2.0) * np.pi))
                term = 0.5 * np.log(sigma2) + ((nu + 1.0) / 2.0) * np.log(1.0 + (eps ** 2) / ((nu - 2.0) * sigma2))
                return term - const
        elif v == "gjr":
            if d == "gaussian":
                omega, alpha, beta, gamma = x
                sigma2 = self._gjr_filter(eps, omega, alpha, beta, gamma)
                return 0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (eps ** 2) / sigma2)
            else:
                omega, alpha, beta, gamma, nu = x
                sigma2 = self._gjr_filter(eps, omega, alpha, beta, gamma)
                const = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log((nu - 2.0) * np.pi))
                term = 0.5 * np.log(sigma2) + ((nu + 1.0) / 2.0) * np.log(1.0 + (eps ** 2) / ((nu - 2.0) * sigma2))
                return term - const
        else:  # egarch, gaussian
            omega, alpha, gamma, beta = x
            sigma2 = self._egarch_filter(eps, omega, alpha, gamma, beta, dist="gaussian")
            return 0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (eps ** 2) / sigma2)

    def _num_hessian(self, f, x: np.ndarray, h: float) -> np.ndarray:
        k = len(x)
        H = np.zeros((k, k), float)
        fx = f(x)
        for i in range(k):
            ei = np.zeros(k); ei[i] = 1.0
            f_ip = f(x + h * ei)
            f_im = f(x - h * ei)
            H[i, i] = (f_ip - 2 * fx + f_im) / (h ** 2)
            for j in range(i + 1, k):
                ej = np.zeros(k)
                ej[j] = 1.0
                f_pp = f(x + h * ei + h * ej)
                f_pm = f(x + h * ei - h * ej)
                f_mp = f(x - h * ei + h * ej)
                f_mm = f(x - h * ei - h * ej)
                hij = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
                H[i, j] = H[j, i] = hij
        return H

    def _compute_inference(self, eps: np.ndarray, xhat: np.ndarray):
        """
        Classical SE: inv(H)
        Robust SE:    inv(H) * (S^T S) * inv(H), where S_t ~= ∂ l_t / ∂θ (numerical)
        """
        h = self.fd_step
        # Total NLL function at theta
        f_total = lambda x: float(np.sum(self._contribs(x, eps)))
        # Hessian
        H = self._num_hessian(f_total, xhat, h)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)

        vcov = H_inv
        se = np.sqrt(np.maximum(np.diag(vcov), 0.0))

        # Robust OPG
        if not self.robust_se:
            return vcov, None, se, None

        # Score_t via finite differences of per-observation contributions
        k = len(xhat)
        T = len(eps)
        S = np.zeros((T, k), float)
        for j in range(k):
            ej = np.zeros(k)
            ej[j] = 1.0
            lp = self._contribs(xhat + h * ej, eps)
            lm = self._contribs(xhat - h * ej, eps)
            S[:, j] = (lp - lm) / (2 * h)
        J = S.T @ S  # OPG
        try:
            vcov_rob = H_inv @ J @ H_inv
        except Exception:
            vcov_rob = None
        se_rob = None if vcov_rob is None else np.sqrt(np.maximum(np.diag(vcov_rob), 0.0))

        return vcov, vcov_rob, se, se_rob
    

# ---------- Helper functions ----------
def ugarch_results_df(res) -> pd.DataFrame:
    """
    Build a tidy DataFrame of parameter estimates and (robust) SEs
    from a UGARCHResult.
    """
    # Preserve parameter order when available
    names = getattr(res, "param_names", None) or list(res.params.keys())
    est = [(name, float(res.params[name])) for name in names]
    df = pd.DataFrame(est, columns=["param", "estimate"]).set_index("param")

    if getattr(res, "se", None) is not None and len(res.se) == len(names):
        df["se"] = res.se
    if getattr(res, "se_robust", None) is not None and len(res.se_robust) == len(names):
        df["se_robust"] = res.se_robust
    return df

def ugarch_summary(res) -> str:
    df = ugarch_results_df(res)
    lines = []
    lines.append("UGARCH Results")
    lines.append("-" * 60)
    lines.append(
        f"dist: {res.dist} | vol: {getattr(res, 'vol', 'garch')} | "
        f"mean: {getattr(res, 'mean', 'constant')} {getattr(res,'mean_order',(0,0,0))}"
    )
    lines.append(f"converged: {res.success} | message: {res.message}")
    lines.append("")
    lines.append(df.to_string(float_format=lambda x: f"{x:.6g}"))
    lines.append("")
    lines.append(f"mu: {res.mu:.6g}")
    return "\n".join(lines)