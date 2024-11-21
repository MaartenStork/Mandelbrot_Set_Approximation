import numpy as np
from scipy.stats.qmc import LatinHypercube, scale, Sobol

def generate_complex_grid(resolution, real_range=(-2, 1), imag_range=(-1.5, 1.5)):
    """
    Generates a grid of complex numbers for the Mandelbrot set.

    Parameters:
    - resolution: Number of points along each axis.
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.

    Returns:
    - C: 2D array of complex numbers representing the grid.
    """
    # Create equally spaced values for the real and imaginary parts
    rval = np.linspace(real_range[0], real_range[1], resolution).reshape(1, -1)
    ival = np.linspace(imag_range[0], imag_range[1], resolution).reshape(-1, 1)
    # Combine real and imaginary parts into complex numbers
    C = rval + 1.j * ival
    return C

class AdaptiveSampler:
    def __init__(self, num_samples, max_iter, real_range=(-2, 1), imag_range=(-1.5, 1.5),
                 seed=None, error_threshold=1e-4, max_iterations=10):
        """
        Initializes the AdaptiveSampler for the Mandelbrot set.

        Parameters:
        - num_samples: Total number of samples to use in the adaptive sampling process.
        - max_iter: Maximum number of iterations for the Mandelbrot calculation.
        - real_range: Tuple specifying the range for the real axis of the complex plane.
        - imag_range: Tuple specifying the range for the imaginary axis of the complex plane.
        - seed: Random seed for reproducibility. Defaults to a random seed if not provided.
        - error_threshold: Convergence threshold for the total variance of the area estimate.
        - max_iterations: Maximum number of adaptive refinement iterations.
        """
        self.num_samples = num_samples
        self.max_iter = max_iter
        self.real_range = real_range
        self.imag_range = imag_range
        self.seed = seed if seed is not None else random.randint(0, 10000)
        self.error_threshold = error_threshold
        self.max_iterations = max_iterations

        np.random.seed(self.seed)

        # Calculate initial grid size and samples per region
        self.initial_grid_size = max(1, int(np.sqrt(self.num_samples)))
        self.initial_samples_per_region = max(1, self.num_samples // (self.initial_grid_size ** 2))
        self.regions = self.create_grid()

    class Region:
        def __init__(self, x_min, x_max, y_min, y_max):
            """
            Represents a subregion in the complex plane.

            Parameters:
            - x_min, x_max: Bounds for the real axis.
            - y_min, y_max: Bounds for the imaginary axis.
            """
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max
            self.samples = []  # Stores sampled complex numbers
            self.in_set_counts = []  # Tracks whether samples are in the Mandelbrot set
            self.total_samples = 0
            self.local_area_estimate = 0  # Current area estimate for this region
            self.local_variance = np.inf  # Variance estimate for this region

        def sample(self, num_samples, max_iter):
            """
            Samples the region and updates the area and variance estimates.

            Parameters:
            - num_samples: Number of new samples to draw.
            - max_iter: Maximum number of iterations for Mandelbrot set computation.
            """
            x_samples = np.random.uniform(self.x_min, self.x_max, num_samples)
            y_samples = np.random.uniform(self.y_min, self.y_max, num_samples)
            complex_samples = x_samples + 1j * y_samples

            # Evaluate whether each sample belongs to the Mandelbrot set
            in_set = np.array([is_in_mandelbrot(c, max_iter) for c in complex_samples])
            self.samples.extend(complex_samples)
            self.in_set_counts.extend(in_set)
            self.total_samples += num_samples

            # Update area and variance estimates
            area = (self.x_max - self.x_min) * (self.y_max - self.y_min)
            p = np.mean(self.in_set_counts)
            self.local_area_estimate = p * area
            self.local_variance = (p * (1 - p) / self.total_samples) * area ** 2 if self.total_samples > 0 else np.inf

        def increase_sample_count(self, num_samples, max_iter):
            """
            Adds more samples to refine the region's area estimate.

            Parameters:
            - num_samples: Number of additional samples.
            - max_iter: Maximum number of iterations for Mandelbrot set computation.
            """
            self.sample(num_samples, max_iter)

    def create_grid(self):
        """
        Divides the complex plane into an initial grid of regions.

        Returns:
        - List of Region objects representing subregions of the complex plane.
        """
        x_min, x_max = self.real_range
        y_min, y_max = self.imag_range
        x_edges = np.linspace(x_min, x_max, self.initial_grid_size + 1)
        y_edges = np.linspace(y_min, y_max, self.initial_grid_size + 1)

        regions = []
        for i in range(self.initial_grid_size):
            for j in range(self.initial_grid_size):
                region = self.Region(x_edges[i], x_edges[i + 1], y_edges[j], y_edges[j + 1])
                regions.append(region)
        return regions

    def combine_region_estimates(self):
        """
        Computes the total area estimate by summing the local estimates from all regions.

        Returns:
        - Total area estimate of the Mandelbrot set within the defined ranges.
        """
        total_area_estimate = sum(region.local_area_estimate for region in self.regions)
        return total_area_estimate

    def estimate_overall_variance(self):
        """
        Computes the total variance across all regions.

        Returns:
        - Combined variance estimate.
        """
        total_variance = sum(region.local_variance for region in self.regions)
        return total_variance

    def select_regions_with_high_variance(self, fraction=0.5):
        """
        Selects regions with the highest variances for refinement.

        Parameters:
        - fraction: Fraction of regions to select based on their variance.

        Returns:
        - List of regions to refine.
        """
        variances = np.array([region.local_variance for region in self.regions])
        threshold = np.percentile(variances, 100 * (1 - fraction))
        regions_to_refine = [region for region in self.regions if region.local_variance >= threshold]
        return regions_to_refine

    def run(self):
        """
        Executes the adaptive sampling process to estimate the Mandelbrot set's area.

        Returns:
        - Final area estimate after convergence or maximum iterations.
        """
        for iteration in range(self.max_iterations):
            # Sample all regions in the grid
            for region in self.regions:
                samples_to_draw = max(1, self.initial_samples_per_region // (iteration + 1)) \
                                  if region.total_samples > 0 else self.initial_samples_per_region
                region.sample(samples_to_draw, self.max_iter)

            # Update area and variance estimates
            area_estimate = self.combine_region_estimates()
            variance_estimate = self.estimate_overall_variance()

            # Check for convergence
            if variance_estimate < self.error_threshold:
                break

            # Refine regions with high variance
            regions_to_refine = self.select_regions_with_high_variance()
            if not regions_to_refine:
                break
            additional_samples = max(1, self.initial_samples_per_region // (iteration + 1))
            for region in regions_to_refine:
                region.increase_sample_count(additional_samples, self.max_iter)

        return area_estimate

def uniform_random_sampling(num_samples, real_range=(-2, 2), imag_range=(-2, 2), seed=42):
    """
    Performs uniform random sampling for the Mandelbrot set.

    Parameters:
    - num_samples: Number of samples.
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.
    - seed: Random seed for reproducibility.

    Returns:
    - C: 1D array of complex numbers representing the samples.
    """
    # Seed the random generator for reproducibility
    np.random.seed(seed)
    # Generate random real and imaginary parts
    rval = np.random.uniform(real_range[0], real_range[1], num_samples)
    ival = np.random.uniform(imag_range[0], imag_range[1], num_samples)
    return rval + 1.j * ival

def orthogonal_sampling(num_samples, real_range=(-2, 1), imag_range=(-1.5, 1.5), seed=42):
    """
    Performs orthogonal sampling for the Mandelbrot set, suited for Python while following the logic of the provided C code.

    Parameters:
    - num_samples: Total number of samples to generate (must be a perfect square).
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.
    - seed: Random seed for reproducibility.

    Returns:
    - C: 1D array of complex numbers representing the samples.
    """
    # Ensure num_samples is a perfect square for orthogonal grid
    np.random.seed(seed)
    major = int(np.round(np.sqrt(num_samples)))
    num_samples = major * major
    print(f"Adjusted num_samples to {num_samples} to ensure it is a perfect square close to the original value.")

    x_indices = np.arange(major)
    y_indices = np.arange(major)

    samples = []
    for i in range(major):
        # Shuffle indices for randomized sampling
        np.random.shuffle(x_indices)
        np.random.shuffle(y_indices)

        for j in range(major):
            # Generate random perturbations within each grid cell
            rand_real = np.random.uniform(0, 1)
            rand_imag = np.random.uniform(0, 1)

            # Map to the real and imaginary ranges
            x = real_range[0] + (real_range[1] - real_range[0]) * ((i + rand_real) / major)
            y = imag_range[0] + (imag_range[1] - imag_range[0]) * ((j + rand_imag) / major)

            samples.append(complex(x, y))

    return np.array(samples)

def latin_hypercube_sampling(num_samples, real_range=(-2, 1), imag_range=(-1.5, 1.5), seed=42):
    """
    Performs Latin Hypercube sampling for the Mandelbrot set.

    Parameters:
    - num_samples: Number of samples.
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.
    - seed: Random seed for reproducibility.

    Returns:
    - C: 1D array of complex numbers representing the samples.
    """
    # Generate Latin Hypercube samples
    np.random.seed(seed)
    sampler = LatinHypercube(d=2)
    lhs_samples = sampler.random(n=num_samples)

    # Scale samples to the desired ranges
    scaled_samples = scale(lhs_samples, [real_range[0], imag_range[0]], [real_range[1], imag_range[1]])
    rval, ival = scaled_samples[:, 0], scaled_samples[:, 1]
    return rval + 1.j * ival

def importance_sampling(num_samples, max_iter, real_range=(-2, 1), imag_range=(-1.5, 1.5), seed=None):
    """
    Performs importance sampling for estimating the area of the Mandelbrot set.

    Parameters:
    - num_samples: Number of samples.
    - max_iter: Maximum number of iterations for Mandelbrot membership.
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.
    - seed: Random seed for reproducibility.

    Returns:
    - weighted_area_estimate: Estimated area of the Mandelbrot set.
    """
    if seed is None:
        seed = random.randint(0, 10000)
    np.random.seed(seed)

    # Generate normally distributed samples
    mean_real = 0
    mean_imag = 0
    std_dev = 0.8
    
    real_samples = np.random.normal(mean_real, std_dev, num_samples)
    imag_samples = np.random.normal(mean_imag, std_dev, num_samples)

    # Clip samples to remain within specified ranges
    real_samples = np.clip(real_samples, real_range[0], real_range[1])
    imag_samples = np.clip(imag_samples, imag_range[0], imag_range[1])

    # Combine into complex numbers
    complex_samples = real_samples + 1j * imag_samples

    # Compute proposal and target densities
    proposal_density = (1 / (2 * np.pi * std_dev**2)) * np.exp(-(real_samples**2 + imag_samples**2) / (2 * std_dev**2))
    target_density = 1 / ((real_range[1] - real_range[0]) * (imag_range[1] - imag_range[0]))

    # Determine if samples are in the Mandelbrot set
    in_set = np.array([is_in_mandelbrot(c, max_iter) for c in complex_samples])

    # Compute weights and estimate area
    weights = target_density / proposal_density
    rect_area = (real_range[1] - real_range[0]) * (imag_range[1] - imag_range[0])
    weighted_area_estimate = np.mean(in_set * weights) * rect_area

    return weighted_area_estimate

def sobol_sampling(num_samples, real_range=(-2, 1), imag_range=(-1.5, 1.5), seed=None):
    """
    Performs Sobol sampling (Quasi-Monte Carlo) for the Mandelbrot set.

    Parameters:
    - num_samples: Number of samples.
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.
    - seed: Random seed for reproducibility.

    Returns:
    - C: 1D array of complex numbers representing the samples.
    """
    if seed is None:
        seed = random.randint(0, 10000)
    np.random.seed(seed)

    # Generate Sobol samples
    sampler = Sobol(d=2, scramble=True)
    sobol_samples = sampler.random(n=num_samples)

    # Scale samples to the desired ranges
    scaled_samples = scale(sobol_samples, [real_range[0], imag_range[0]], [real_range[1], imag_range[1]])

    rval, ival = scaled_samples[:, 0], scaled_samples[:, 1]
    return rval + 1.j * ival

def pure_random_sampling(num_samples, real_range=(-2, 1), imag_range=(-1.5, 1.5), seed=42):
    """
    Performs pure random sampling for the Mandelbrot set.

    Parameters:
    - num_samples: Number of samples.
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.
    - seed: Random seed for reproducibility.

    Returns:
    - C: 1D array of complex numbers representing the samples.
    """
    np.random.seed(seed)
    # Generate uniformly distributed samples
    rval = np.random.uniform(real_range[0], real_range[1], num_samples)
    ival = np.random.uniform(imag_range[0], imag_range[1], num_samples)
    return rval + 1.j * ival