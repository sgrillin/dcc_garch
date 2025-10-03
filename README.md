# dcc_garch

*A lightweight NumPy/SciPy implementation of Dynamic Conditional Correlation (DCC)–GARCH for multivariate returns (n > 2).*

![status](https://img.shields.io/badge/status-experimental-blue)
![python](https://img.shields.io/badge/python-3.9%2B-informational)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **DCC(1,1)** correlation dynamics with **Gaussian** or **Student-t** likelihood
- Optional **ADCC** asymmetry term (downside correlation spikes via parameter `g`)
- Per-asset **UGARCH(1,1)** (Gaussian or Student-t QMLE)
- **Rolling / Expanding** window estimator
- Simple, consistent API (`fit/forecast`, `get_params/set_params`)
- Returns full paths: `Q_t`, `R_t`, `D_t`, and `H_t = D_t R_t D_t`

---

## Installation

```bash
# dependencies
pip install -U numpy scipy pandas

# install this package (editable mode recommended for development)
pip install -e .
