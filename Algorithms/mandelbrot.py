import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_mandelbrot(C, max_steps=100, bound=2, power=2, area_factor=9):
    """
    Computes the Mandelbrot set using the standard iteration method

    Parameters:
    - C: ndarray of complex numbers.
    - max_steps: Maximum number of iterations.
    - bound: Divergence threshold.
    - power: Power used in the Mandelbrot calculation.
    - area_factor: Factor for estimating the set area.

    Returns:
    - num_div_steps: 2D array of the number of steps before divergence.
    - area_at_step: list of the estimated area at each iteration.
    """
    # Initialize the iteration variable Z to the initial values of C
    Zi = C.copy()
    # Create an array to store the number of steps before divergence
    num_div_steps = np.zeros_like(C, dtype=np.int64)
    # List to keep track of estimated area at each iteration step
    area_at_step = []

    # Iterate up to the maximum number of steps
    for i in range(max_steps):
        # Apply the Mandelbrot formula: Z = Z^power + C
        Zi = np.power(Zi, power) + C
        # Update the number of steps before divergence for points still within the bound
        num_div_steps += (np.abs(Zi) < bound)
        
        # Estimate the relative area of points still within the bound
        rel_area = np.sum(np.abs(Zi) < bound) / np.prod(C.shape)
        # Scale the relative area using the area factor and store it
        area_at_step.append(rel_area * area_factor)

    # Print the final area estimation for debugging or analysis
    print(f"Final area estimation: {area_at_step[-1]}")

    return num_div_steps, np.array(area_at_step)


def compute_mandelbrot_torch(C, max_steps=100, bound=2, power=2, area_factor=9):
    """
    Computes the Mandelbrot set using the standard iteration method
    Uses pytorch methods instead of numpy for better scaling on multiple processors / GPU

    Parameters:
    - C: ndarray of complex numbers or torch tensor.
    - max_steps: Maximum number of iterations.
    - bound: Divergence threshold.
    - power: Power used in the Mandelbrot calculation.
    - area_factor: Factor for estimating the set area.

    Returns:
    - num_div_steps: 2D array of the number of steps before divergence.
    - area_at_step: list of the estimated area at each iteration.
    """
    # Convert input to a torch tensor if it isn't already
    if not torch.is_tensor(C):
        C = torch.from_numpy(C)

    # Number of elements in the input tensor
    num_samples = C.numel()
    # Clone the input tensor to initialize the iteration variable Z
    Zi = C.detach().clone()
    # Tensor to store the number of steps before divergence
    num_div_steps = torch.zeros_like(C, dtype=torch.int64)
    # List to store estimated areas at each iteration step
    area_at_step = []

    # Iterate up to the maximum number of steps
    for i in range(max_steps):
        # Apply the Mandelbrot formula: Z = Z^power + C
        Zi = torch.pow(Zi, power) + C
        # Update the divergence steps for points still within the bound
        num_div_steps += (torch.abs(Zi) < bound)
        
        # Estimate the relative area of points still within the bound
        rel_area = torch.sum(torch.abs(Zi) < bound) / num_samples
        # Scale the relative area using the area factor and store it
        area_at_step.append(rel_area * area_factor)

    # Print the final area estimation for debugging or analysis
    print(f"Final area estimation: {area_at_step[-1]}")

    return num_div_steps, torch.tensor(area_at_step)

