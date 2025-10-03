# dcc_garch (v2)

A lightweight, NumPy/SciPy-based implementation of **DCC(1,1)-GARCH** for multivariate return series (n > 2), with:

- **Student-t** innovations in univariate GARCH (optional, estimate `nu`)
- **Gaussian or Student-t DCC** correlation likelihood (`dist="gaussian"` or `"student"`)
- **ADCC** asymmetry term (`asym=True`) with parameter `g`
- **Rolling / Expanding** window estimator (`RollingDCC`)
- **statsmodels/sklearn-style API** (`get_params`, `set_params`; `fit` can return `self`)

## Install

```bash
pip install -U numpy scipy pandas
pip install -e .
```

## Quick start

```python
import numpy as np
from dcc_garch import DCC

np.random.seed(0)
T, N = 1000, 4
X = 0.001 + 0.01*np.random.randn(T, N)

# Gaussian DCC
dcc = DCC(mean="constant", dist="gaussian", asym=False)
res = dcc.fit(X)
print(res.a, res.b, res.corr_half_life)

# Student-t DCC with asymmetry
dcc_t = DCC(mean="constant", dist="student", asym=True)
res_t = dcc_t.fit(X)
print("a,b,g,nu:", res_t.a, res_t.b, res_t.g, res_t.nu)

fc = dcc.forecast(steps=1)
print(fc["H"][0].shape)   # (N, N)
```

## Rolling / Expanding

```python
from dcc_garch import RollingDCC

roll = RollingDCC(window=500, step=50, expanding=False, mean="constant", dist="gaussian", asym=False)
roll.fit(X)
H_last = roll.last_covariances()  # (num_windows, N, N)
R_last = roll.last_correlations()
```

## Univariate UGARCH (optional Student-t)

```python
from dcc_garch import UGARCH

u = UGARCH(mean="constant", dist="student")
ures = u.fit(X[:,0])
print(ures.params)  # includes nu when dist="student"
```

## Notes

- Constraints: 
  - univariate: `omega>0`, `alpha>=0`, `beta>=0`, `alpha+beta<1`.
  - DCC: `a>=0`, `b>=0`; with `asym=True`, we enforce `a+b+g/2<1` (heuristic for persistence).
- For large N/noisy data, consider shrinking `Qbar` externally.
- Forecasts keep variances at last `D_t`; for fully consistent forward paths, plug in your own univariate forecasts.

MIT License.
