# Copilot Instructions for dcc_garch

This guide enables AI coding agents to be productive in the `dcc_garch` codebase. It summarizes architecture, workflows, and conventions specific to this project.

## Project Overview
- Implements Dynamic Conditional Correlation (DCC)-GARCH models for multivariate financial returns.
- Core logic is in `dcc_garch/`:
  - `dcc.py`: Multivariate DCC(1,1)-GARCH, ADCC, Gaussian/Student-t likelihoods, main API (`DCC` class, `fit`, `forecast`).
  - `univariate.py`: Univariate GARCH, GJR-GARCH, EGARCH models (`UGARCH` class), used per-asset in DCC.
  - `utils.py`: Array utilities, matrix operations, numerical stability helpers.
- Example usage in `examples/example_usage.py`.

## Key Patterns & API
- Main entrypoint: `DCC` class (`dcc.py`). Usage: `fit(X)`, `forecast(steps)`, access results via attributes or dicts.
- Per-asset volatility handled by `UGARCH` (from `univariate.py`).
- Rolling/expanding window estimation via `RollingDCC` (see example).
- Results returned as dataclasses (`DCCResult`, `UGARCHResult`) with all relevant parameters and paths.
- All models support both Gaussian and Student-t likelihoods. Asymmetry (ADCC) is optional via `asym=True`.
- API is designed for batch/array operations (NumPy, pandas).

## Developer Workflows
- **Install dependencies:**
  ```bash
  pip install -U numpy scipy pandas
  pip install -e .
  ```
- **Run examples:**
  ```bash
  python examples/example_usage.py
  ```
- **Testing:** No formal test suite detected; validate via example script or add tests in `examples/`.
- **Debugging:** Use print statements or inspect returned dataclasses for model internals.

## Conventions & Integration
- All code is type-annotated and uses dataclasses for results.
- Optional dependencies (SciPy, statsmodels) are handled with fallbacks for missing packages.
- Matrix operations use ridge regularization for stability (`ridge_pd`, `safe_invert`).
- Parameters and results are always accessible via `.get_params()`, `.result_`, or returned objects.
- No external service integration; all computation is local and batch-oriented.

## Examples
- See `examples/example_usage.py` for typical usage patterns:
  - Fit DCC model: `dcc = DCC(...); res = dcc.fit(X)`
  - Forecast: `dcc.forecast(steps=1)`
  - Rolling estimation: `RollingDCC(...).fit(X)`

## File Reference
- `dcc_garch/dcc.py`: DCC model, main API
- `dcc_garch/univariate.py`: Univariate GARCH models
- `dcc_garch/utils.py`: Matrix/array helpers
- `examples/example_usage.py`: Usage patterns

---

If any section is unclear or missing, please provide feedback for further refinement.