# dualTIRF_Analysis

## Overview
`dualTIRF_Analysis.py` is a Python script designed for analyzing trajectories from dual Total Internal Reflection Fluorescence (TIRF) microscopy data. The script processes CSV files containing track and spot statistics, calculates various metrics such as Mean Squared Displacement (MSD), diffusion coefficients, and intensity values, and generates plots and summary statistics.

## Features
- **Trajectory Analysis**: Calculates MSD, diffusion coefficients, alpha values, and sMSS for trajectories.
- **Intensity Analysis**: Computes mean intensity values and background-subtracted intensity for specified spot ranges.
- **Overlap Detection**: Identifies overlapping tracks between two channels (Rojo and Verde) based on spatial and temporal proximity.
- **Plot Generation**: Creates plots for position vs. frame and intensity vs. frame for overlapping tracks.
- **CSV Processing**: Updates CSV files with new columns and saves results to separate files.
- **Parallel Processing**: Utilizes `ThreadPoolExecutor` for parallel analysis of trajectories.

## Dependencies
- `os`
- `pandas`
- `concurrent.futures`
- `matplotlib`
- `mpl_toolkits.mplot3d`
- `numpy`
- `skimage`
- `PIL`
- `decimal`

## Installation
To use this script, ensure you have the required dependencies installed. You can install them using pip:

```bash
pip install pandas matplotlib numpy scikit-image pillow
