# dcc_garch

*A lightweight, research-grade Python implementation of Dynamic Conditional Correlation (DCC)–GARCH models for multivariate financial returns.*

![status](https://img.shields.io/badge/status-stable-blue)
![python](https://img.shields.io/badge/python-3.8%2B-informational)
![license](https://img.shields.io/badge/license-MIT-green)
![build](https://img.shields.io/badge/build-passing-success)

---

## 🧭 Overview

The `dcc_garch` package implements the **Dynamic Conditional Correlation (DCC)** framework proposed by **Engle (2002)** — a cornerstone in multivariate volatility modeling.  
It estimates time-varying covariance and correlation matrices by combining univariate GARCH-type models with a dynamic correlation structure.

This implementation is designed for **clarity, extensibility, and research replication**.  
It supports **Gaussian** and **Student–t** innovations, **asymmetric correlation** (ADCC), and a variety of univariate volatility and mean specifications including **ARMA**, **ARIMA**, **GJR–GARCH**, and **EGARCH**.

---

## ✨ Features

- **Univariate step (Stage 1):**
  - Constant, **ARMA(p,q)**, or **ARIMA(p,d,q)** mean structures  
  - **GARCH(1,1)**, **GJR–GARCH**, and **EGARCH** volatility dynamics  
  - Gaussian or Student–t quasi–maximum likelihood (QMLE)  

- **Multivariate step (Stage 2):**
  - **DCC(1,1)** and **ADCC(1,1)** correlation dynamics  
  - Estimation via **L-BFGS-B** optimization with ridge stabilization  
  - Full log-likelihood under Gaussian or Student–t distributions  

- **Advanced options:**
  - Rolling and expanding window estimation (`RollingDCC`)  
  - Parameter inference with robust (sandwich) standard errors  
  - Forecasts of conditional covariance and correlation matrices  
  - Modular class design (`UGARCH`, `DCC`, `RollingDCC`)  

---

## 🧩 Installation

```bash
# dependencies
pip install -U numpy scipy pandas statsmodels matplotlib

# install this package (editable mode recommended for development)
pip install -e .
