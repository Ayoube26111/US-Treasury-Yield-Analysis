# US-Treasury-Yield-Analysis

**Author:** Ayoub Elfilali  
**Date:** 2026-03-15  

---

## Overview
This repository contains a comprehensive analysis of U.S. Treasury yield data spanning 1990–2020. The project includes:

- Dataset investigation and metadata extraction
- Yield curve slope and shape analysis (2-Year vs 10-Year yields)
- Nelson-Siegel yield curve fitting
- Principal Component Analysis (PCA) to extract Level, Slope, and Curvature factors

The goal is to explore historical Treasury yields, understand the shape of the yield curve at specific dates, and decompose yield movements into interpretable components for research or trading applications.

---

## Repository Contents

| File | Description |
|------|-------------|
| `Treasury_Yield_Analysis.ipynb` | Main Jupyter notebook containing code, explanations, and plots. |
| `utils.py` | Helper functions: data inspection, yield curve analysis, NS fitting, PCA. |
| `treasury_yields.csv` | Raw U.S. Treasury yield dataset (ensure downloaded or provided). |

---

## Requirements

- Python ≥3.10  
- Packages:

```bash
pip install pandas numpy matplotlib scipy
