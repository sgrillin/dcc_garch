"""
Basic demo for DCC-GARCH.

- Simulates a T x N return matrix
- Fits Student-t DCC with asymmetry
- Forecasts one step ahead
- Runs a rolling (windowed) Gaussian DCC
"""

import numpy as np
from dcc_garch import DCC, RollingDCC


if __name__ == "__main__":
        np.random.seed(42)
        T, N = 800, 5
        # simple synthetic returns (zero cross-corr, just for demo)
        X = 0.0005 + 0.01 * np.random.randn(T, N)

        # --- Fit: Student-t DCC with asymmetry (ADCC) ---
        dcc = DCC(mean="constant", dist="student", asym=True)
        res = dcc.fit(X)
        print("=== DCC fit (Student-t, asym) ===")
        print(f"Estimated a, b, g, nu: {res.a:.4f}, {res.b:.4f}, {res.g:.4f}, {res.nu:.2f}")
        print(f"Correlation half-life: {res.corr_half_life:.2f}")

        # --- 1-step forecast of Q, R, H ---
        fc = dcc.forecast(steps=1)
        print("\n=== 1-step forecast ===")
        print("Q forecast shape:", fc["Q"].shape)  # (1, N, N)
        print("R forecast shape:", fc["R"].shape)  # (1, N, N)
        print("H forecast shape:", fc["H"].shape)  # (1, N, N)

        # --- Rolling windows example (Gaussian, no asymmetry) ---
        print("\n=== Rolling DCC (Gaussian) ===")
        roll = RollingDCC(
            window=400, step=50, expanding=False,
            mean="constant", dist="gaussian", asym=False
        )
        roll.fit(X)
        print("Num windows:", len(roll.results_))
        print("Last R shape per window:", roll.last_correlations().shape)  # (num_windows, N, N)
        print("Last H shape per window:", roll.last_covariances().shape)   # (num_windows, N, N)")

