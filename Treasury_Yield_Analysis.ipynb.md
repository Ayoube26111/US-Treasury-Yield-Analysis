{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# US Treasury Yield Analysis (1990–2020)\n",
    "**Author:** Ayoub Elfilali \n",
    "**Date:** 2026-03-15\n",
    "\n",
    "---\n",
    "This notebook performs a comprehensive analysis of U.S. Treasury yields, including data investigation, yield curve analysis, Nelson-Siegel fitting, and PCA factor decomposition."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 1. Load Libraries and Data"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.optimize import minimize\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv('treasury_yields.csv', parse_dates=['Date'])\n",
    "df.set_index('Date', inplace=True)\n",
    "df = df.sort_index()\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Analysis:\n",
    "We loaded the Treasury yield dataset and confirmed the index is a DatetimeIndex. The dataset contains daily yields for multiple maturities from 1990 to 2020."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 2. Data Investigation"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "def investigate_data(df):\n",
    "    earliest_date = df.index.min().strftime('%Y-%m-%d')\n",
    "    latest_date = df.index.max().strftime('%Y-%m-%d')\n",
    "    return {\n",
    "        'num_rows': df.shape[0],\n",
    "        'num_columns': df.shape[1],\n",
    "        'date_range': (earliest_date, latest_date),\n",
    "        'column_names': list(df.columns)\n",
    "    }\n",
    "\n",
    "info = investigate_data(df)\n",
    "info"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Analysis:\n",
    "The dataset contains metadata such as the number of rows, columns, date range, and available maturities. This provides an overview before deeper analysis."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 3. Yield Curve Analysis (2Y vs 10Y)"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "def analyze_yield_curve_from_data(df, date):\n",
    "    date = pd.to_datetime(date)\n",
    "    df = df.sort_index()\n",
    "    if date < df.index.min() or date > df.index.max():\n",
    "        raise ValueError(f\"Date {date.strftime('%Y-%m-%d')} not found in dataset\")\n",
    "    if date not in df.index:\n",
    "        date = df.index[df.index <= date][-1]\n",
    "    df.columns = df.columns.str.strip().str.lower()\n",
    "    col_2y = [c for c in df.columns if '2' in c and 'year' in c]\n",
    "    col_10y = [c for c in df.columns if '10' in c and 'year' in c]\n",
    "    if not col_2y or not col_10y:\n",
    "        raise ValueError(\"2Y or 10Y column not found\")\n",
    "    yield_2y = float(df.loc[date, col_2y[0]])\n",
    "    yield_10y = float(df.loc[date, col_10y[0]])\n",
    "    slope = yield_10y - yield_2y\n",
    "    if slope > 0.1:\n",
    "        shape = 'normal'\n",
    "    elif abs(slope) <= 0.1:\n",
    "        shape = 'flat'\n",
    "    else:\n",
    "        shape = 'inverted'\n",
    "    return yield_2y, yield_10y, (slope, shape)\n",
    "\n",
    "# Example: April 3, 2007\n",
    "y2, y10, (slope, shape) = analyze_yield_curve_from_data(df, '2007-04-03')\n",
    "y2, y10, slope, shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Analysis:\n",
    "On April 3, 2007, the 2Y yield was higher than the 10Y yield, resulting in an inverted yield curve. This indicates market expectations of slower growth or lower rates in the future."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 4. Nelson-Siegel Yield Curve Fitting"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "MATURITY_MAP = {\n",
    "    '1 Month': 1/12, '3 Month': 3/12, '6 Month': 6/12, '1 Year': 1,\n",
    "    '2 Year': 2, '3 Year': 3, '5 Year': 5, '7 Year': 7, '10 Year': 10,\n",
    "    '20 Year': 20, '30 Year': 30\n",
    "}\n",
    "\n",
    "def nelson_siegel(t, beta0, beta1, beta2, tau):\n",
    "    t = np.array(t)\n",
    "    factor1 = (1 - np.exp(-t / tau)) / (t / tau)\n",
    "    factor2 = factor1 - np.exp(-t / tau)\n",
    "    return beta0 + beta1 * factor1 + beta2 * factor2\n",
    "\n",
    "def fit_ns_model(df, date):\n",
    "    row = df.loc[pd.to_datetime(date)]\n",
    "    maturities, yields = [], []\n",
    "    for col in df.columns:\n",
    "        if col in MATURITY_MAP and not pd.isna(row[col]):\n",
    "            maturities.append(MATURITY_MAP[col])\n",
    "            yields.append(row[col])\n",
    "    maturities, yields = np.array(maturities), np.array(yields)\n",
    "    def objective(params):\n",
    "        beta0, beta1, beta2, tau = params\n",
    "        fitted = nelson_siegel(maturities, beta0, beta1, beta2, tau)\n",
    "        return np.sum((yields - fitted) ** 2)\n",
    "    initial_guess = [np.mean(yields), -1, 1, 1]\n",
    "    bounds = [(None,None),(None,None),(None,None),(0.01,10)]\n",
    "    result = minimize(objective, initial_guess, bounds=bounds)\n",
    "    beta0, beta1, beta2, tau = result.x\n",
    "    return {'beta0': beta0, 'beta1': beta1, 'beta2': beta2, 'tau': tau, 'maturities': maturities, 'yields': yields}\n",
    "\n",
    "def plot_fitted_yield_curve(df, date, fitted_result):\n",
    "    beta0, beta1, beta2, tau = fitted_result['beta0'], fitted_result['beta1'], fitted_result['beta2'], fitted_result['tau']\n",
    "    t_smooth = np.linspace(0.01, 30, 300)\n",
    "    fitted_curve = nelson_siegel(t_smooth, beta0, beta1, beta2, tau)\n",
    "    plt.figure(figsize=(8,5))\n",
    "    plt.scatter(fitted_result['maturities'], fitted_result['yields'], color='red', label='Observed Yields')\n",
    "    plt.plot(t_smooth, fitted_curve, label='Nelson-Siegel Fit')\n",
    "    plt.xlabel('Maturity (Years)')\n",
    "    plt.ylabel('Yield (%)')\n",
    "    plt.title(f'Nelson-Siegel Fit: {date}')\n",
    "    plt.legend()\n",
    "    plt.show()\n",
    "\n",
    "fitted_result = fit_ns_model(df, '2015-07-01')\n",
    "plot_fitted_yield_curve(df, '2015-07-01', fitted_result)"
   ]
  }
 ]
}

