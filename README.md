# Approximating the Area of the Mandelbrot Set Using Monte Carlo Estimation

## Overview
This repository implements Monte Carlo estimation techniques to approximate the area of the Mandelbrot set. The project uses different sampling methods, including random sampling, orthogonal sampling, and adaptive sampling, to estimate the area and compare their effectiveness. The repository includes the core iteration algorithms for generating Mandelbrot points, sampling methods, and tools for statistical analysis and visualization.

The code focuses on evaluating the accuracy, convergence rates, and computational efficiency of each sampling method. Results and visualizations are provided to illustrate the relative performance of these methods.


## Authors:
- Maarten Stork - 15761770
- Paul Jungnickel - 15716554
- Lucas Keijzer - 14041073

## Files:
- `MaartenStork_15761770_PaulJungnick…5716554_LucasKeijzer_14041073..ipynb` - Notebook containing plotting functions and results
- `mandelbrot.py` - Contains the Mandelbrot iteration algorithms
- `sampling_methods.py` - Contains methods for generating samples from the complex plane
- `GoogleColab.ipynb` - Contains the code from the Google Colab, this code can largly be ignored.
- `/plots` - Contains all the plots generated 

## Dependencies:
- **os**: Used for handling file paths and directory operations.
- **random**: For generating random numbers and seeds.
- **numpy** (`np`): Used for numerical operations, creating grids, and random number generation.
- **pandas** (`pd`): For data manipulation and analysis.
- **matplotlib** (`plt`): Used for plotting graphs and visualizations.
- **scipy.stats**: Used for statistical analysis, such as t-tests.
- **mpl_toolkits.mplot3d** (`Axes3D`): Used for creating 3D plots.
