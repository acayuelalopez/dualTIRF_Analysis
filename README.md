# dualTIRF_Analysis

## Overview

`dualTIRF_Analysis.py` is a Python script designed for analyzing trajectories and intensity values from dual-color Total Internal Reflection Fluorescence (TIRF) microscopy data. The script processes CSV files containing track data, identifies overlapping trajectories, calculates various metrics, and generates plots and summary statistics.

## Features

- **Plotting Overlapping Tracks**: Generates plots for the mean intensity of two different tracks (Rojo and Verde) against the frame number, including background intensity values.
- **Mean Squared Displacement (MSD) Calculations**: Computes MSD, alpha values, and diffusion coefficients for trajectories.
- **Trajectory Analysis**: Processes CSV files, identifies overlapping trajectories, and calculates metrics such as MSD, alpha values, diffusion coefficients, and mean intensities.
- **CSV File Handling**: Locates and reads CSV files, updates them with new columns, and extracts specific column values for summary analysis.
- **Parallel Processing**: Analyzes trajectories in parallel to improve efficiency.
- **Directory Management**: Creates directories for storing results, plots, and summary analyses.
- **Summary Data Generation**: Compiles summary data for overlapping and non-overlapping tracks and saves it to CSV files.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-image
- Pillow

## Installation

1. Clone the repository or download the script.
2. Install the required Python packages using pip:
   ```bash
   pip install pandas numpy matplotlib scikit-image Pillow
   ```
