import numpy as np
from dcc_garch import DCC, RollingDCC

np.random.seed(42)
T, N = 800, 5
X = 0.0005 + 0.01 * np.random.randn(T, N)

# Student-t DCC with asymmetry
dcc = DCC(mean="constant", dist="student", asym=True)
res = dcc.fit(X)
print("Estimated a, b, g, nu:", res.a, res.b, res.g, res.nu)
print("Corr half-life:", res.corr_half_life)

# Forecast 1-step
fc = dcc.forecast(steps=1)
print("H forecast shape:", fc["H"].shape)

# Rolling windows (Gaussian)
roll = RollingDCC(window=400, step=50, expanding=False, mean="constant", dist="gaussian", asym=False)
roll.fit(X)
print("Num windows:", len(roll.results_))
print("Last R shape (per window):", roll.last_correlations().shape)
