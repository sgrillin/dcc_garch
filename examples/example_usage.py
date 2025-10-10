"""
Basic demo for DCC-GARCH.

- Simulates a T x N return matrix
- Fits Student-t DCC with asymmetry
- Prints parameter summaries / tables
- Forecasts one step ahead
- Plots a correlation time series and a final correlation heatmap
"""

import numpy as np
import pandas as pd
from dcc_garch import DCC
import matplotlib.pyplot as plt

# Optional reporting helpers (if you added them to your package)
try:
    from dcc_garch.reporting import dcc_summary, dcc_results_df, ugarch_summary, ugarch_results_df
except Exception:
    dcc_summary = dcc_results_df = ugarch_summary = ugarch_results_df = None


if __name__ == "__main__":
    # --- Generate synthetic data ---
    np.random.seed(42)
    T, N = 800, 5
    X = 0.0005 + 0.01 * np.random.randn(T, N)

    # --- Fit: Student-t DCC with asymmetry (ADCC) ---
    dcc = DCC(mean="constant", dist="student", asym=True)
    res = dcc.fit(X)

    # Headline numbers
    print("=== DCC fit (Student-t, asym) ===")
    print(f"Estimated a, b, g, nu: {res.a:.4f}, {res.b:.4f}, {res.g:.4f}, {res.nu:.2f}")
    print(f"Correlation half-life: {res.corr_half_life:.2f}")

    # Full DCC summary / table (uses helpers if present; otherwise falls back)
    print("\n=== DCC Results ===")
    if hasattr(res, "summary"):
        print(res.summary())
    elif dcc_summary is not None:
        print(dcc_summary(res))
    else:
        print(f"dist: {res.dist} | asym: {res.g is not None}")
        print(f"converged: {res.success} | message: {res.message}")
        core = f"a={res.a:.6f}  b={res.b:.6f}"
        if res.g is not None: core += f"  g={res.g:.6f}"
        if res.nu is not None: core += f"  nu={res.nu:.4f}"
        print(core)
        print(f"persistence={res.persistence:.6f} | corr half-life={res.corr_half_life:.2f}")

    # Parameter table (if available)
    df_params = None
    if hasattr(res, "to_frame"):
        df_params = res.to_frame()
    elif dcc_results_df is not None:
        df_params = dcc_results_df(res)
    if isinstance(df_params, pd.DataFrame):
        print("\nDCC parameters table:")
        print(df_params.to_string(float_format=lambda x: f"{x:.6g}"))

    # Show a couple of univariate GARCH results that fed into DCC (if present)
    if getattr(res, "ugarch_params", None):
        print("\n=== Univariate GARCH (first 2 series) ===")
        for i, p in enumerate(res.ugarch_params[:2]):
            print(f"\n[Series {i}] params:")
            for k in ("omega", "alpha", "beta", "gamma", "nu"):
                if k in p:
                    print(f"  {k:>6s}: {p[k]:.6g}")

    # --- 1-step forecast of Q, R, H ---
    fc = dcc.forecast(steps=1)
    print("\n=== 1-step forecast ===")
    print("Q forecast shape:", fc["Q"].shape)  # (1, N, N)
    print("R forecast shape:", fc["R"].shape)  # (1, N, N)
    print("H forecast shape:", fc["H"].shape)  # (1, N, N)

    # --- Simple plots (if matplotlib is installed) ---
    if _HAVE_MPL:
        R_t = res.R_t  # (T, N, N)

        # 1) Time series of dynamic correlation for a pair (0,1)
        plt.figure()
        plt.plot(R_t[:, 0, 1])
        plt.title("Dynamic correlation: assets (0,1)")
        plt.xlabel("time")
        plt.ylabel("rho_01")
        plt.tight_layout()

        # 2) Heatmap of the last correlation matrix
        plt.figure()
        im = plt.imshow(R_t[-1], vmin=-1, vmax=1, interpolation="nearest")
        plt.colorbar(im, label="corr")
        plt.title("Final correlation matrix $R_T$")
        plt.xticks(range(N), [f"{i}" for i in range(N)])
        plt.yticks(range(N), [f"{i}" for i in range(N)])
        plt.tight_layout()

        plt.show()
    else:
        print("\n(matplotlib not installed; skipping plots)")

# %%