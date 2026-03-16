 README.md

```markdown
# Treasury Yield Analysis

**Author:** Your Name  
**Date:** 2026-03-15  

## Overview
This repository contains a comprehensive analysis of U.S. Treasury yield data from 1990–2020. The analysis includes:

- Dataset investigation and metadata extraction
- Yield curve slope and shape analysis (2Y–10Y)
- Nelson-Siegel curve fitting
- Principal Component Analysis for Level, Slope, and Curvature factors

## Files
- `Treasury_Yield_Analysis.ipynb`: Jupyter notebook with all analysis
- `utils.py`: Helper functions for data inspection, yield curve analysis, NS fitting, and PCA
- `treasury_yields.csv`: Raw Treasury yield dataset (add your CSV here)

## Requirements
- Python ≥3.10
- pandas, numpy, matplotlib, scipy

Install dependencies:

```bash
pip install pandas numpy matplotlib scipy
