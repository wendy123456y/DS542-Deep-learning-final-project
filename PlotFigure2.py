"""

This script performs the following major steps:
1. Load posterior samples
2. Load ground truth statistics
3. Reverse standardization to physical units
4. Prepare a radius grid for evaluation
5. Define physical model functions: density, mass, and anisotropy
6. Preprocess standardized samples into model parameters
7. Summarize posterior samples by computing median and credible intervals
8. Generate summaries for both cored and cuspy profile samples
9. Compute ground truth curves from published parameter values
10. Plot the profile summaries against ground truth for comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# === Step 1: Load posterior samples ===
samples_cored_std = np.load('posterior_samples.npy').squeeze(0)   # shape (10000, 5)
samples_cuspy_std = np.load('posterior_samples_cuspy.npy').reshape(-1, 5)  # shape (31800, 5)

# === Step 2: Load ground truth statistics ===
df = pd.read_csv('data/ground_truths.csv')
means = df.mean()
stds = df.std()

# === Step 3: Reverse standardization ===
samples_cored_phys = samples_cored_std * stds.values + means.values
samples_cuspy_phys = samples_cuspy_std * stds.values + means.values

# === Step 4: Prepare radius grid ===
r = np.geomspace(0.01, 10, 100)
log_r = np.log10(r)

# === Step 5: Define physical model functions ===
def rho_func(r, rho0, rs, gamma):
    """
    Compute the density profile ρ(r) for given parameters.

    Args:
        r (array-like): Radius values.
        rho0 (float): Scale density.
        rs (float): Scale radius.
        gamma (float): Inner slope parameter.

    Returns:
        array-like: Density values ρ(r).
    """
    return rho0 * (r/rs)**(-gamma) * (1 + r/rs)**(gamma - 4)

def mass_func(r, rho0, rs, gamma):
    """
    Compute the enclosed mass profile M(r) via numerical integration.

    Args:
        r (array-like): Radius values.
        rho0 (float): Scale density.
        rs (float): Scale radius.
        gamma (float): Inner slope parameter.

    Returns:
        array-like: Cumulative mass values M(r).
    """
    integrand = 4 * np.pi * r**2 * rho_func(r, rho0, rs, gamma)
    return cumulative_trapezoid(integrand, r, initial=0)

def beta_func(r, rstar, ra):
    """
    Compute the anisotropy parameter β(r) for orbital distribution.

    Args:
        r (array-like): Radius values.
        rstar (float): Stellar scale radius.
        ra (float): Anisotropy radius.

    Returns:
        array-like: Anisotropy values β(r).
    """
    return r**2 / (r**2 + ra**2)

# === Step 6: Preprocess samples (fix columns) ===
def preprocess(samples_phys):
    """
    Extract and transform raw posterior samples into physical parameters.

    Args:
        samples_phys (ndarray): Array of shape (N,5) with columns
            [log10(rho0), gamma, rs, log10(rstar), log10(ra)].

    Returns:
        tuple: (rho0, rs, gamma, rstar, ra) arrays.
    """
    log_rho0 = samples_phys[:, 0]
    gamma = samples_phys[:, 1]
    rs = samples_phys[:, 2]
    log_rstar = samples_phys[:, 3]
    log_ra = samples_phys[:, 4]

    rho0 = 10**(log_rho0)
    rstar = 10**(log_rstar)
    ra = 10**(log_ra)

    return rho0, rs, gamma, rstar, ra

# === Step 7: Summarize posterior samples ===
def summarize(samples_phys):
    """
    Compute median and 16/84 percentile bands for ρ(r), M(r), and β(r).

    Args:
        samples_phys (ndarray): Physical-parameter samples (N x 5).

    Returns:
        tuple: Nine arrays in order:
            rho_median, rho_low, rho_high,
            mass_median, mass_low, mass_high,
            beta_median, beta_low, beta_high
    """
    rho0, rs, gamma, rstar, ra = preprocess(samples_phys)

    rho_samples = []
    mass_samples = []
    beta_samples = []

    for i in range(samples_phys.shape[0]):
        rho_arr = rho_func(r, rho0[i], rs[i], gamma[i])
        mass_arr = mass_func(r, rho0[i], rs[i], gamma[i])
        beta_arr = beta_func(r, rstar[i], ra[i])

        rho_samples.append(np.log10(rho_arr))
        mass_samples.append(np.log10(mass_arr + 1e-8))  # avoid log(0)
        beta_samples.append(beta_arr)

    rho_samples = np.array(rho_samples)
    mass_samples = np.array(mass_samples)
    beta_samples = np.array(beta_samples)

    rho_median = np.median(rho_samples, axis=0)
    rho_low = np.percentile(rho_samples, 16, axis=0)
    rho_high = np.percentile(rho_samples, 84, axis=0)

    mass_median = np.median(mass_samples, axis=0)
    mass_low = np.percentile(mass_samples, 16, axis=0)
    mass_high = np.percentile(mass_samples, 84, axis=0)

    beta_median = np.median(beta_samples, axis=0)
    beta_low = np.percentile(beta_samples, 16, axis=0)
    beta_high = np.percentile(beta_samples, 84, axis=0)

    return (rho_median, rho_low, rho_high,
            mass_median, mass_low, mass_high,
            beta_median, beta_low, beta_high)

# === Step 8: Summarize cored and cuspy galaxies ===
summary_cored = summarize(samples_cored_phys)
summary_cuspy = summarize(samples_cuspy_phys)

# === Step 9: Generate ground truth curves ===
def get_truth(gamma_truth):
    """
    Compute theoretical ground-truth profiles for a given γ.

    Args:
        gamma_truth (float): Inner slope value (e.g., 0 for cored, 1 for cuspy).

    Returns:
        tuple: (rho_truth, mass_truth, beta_truth) arrays on the grid r.
    """
    rho0_truth = 10**(-0.1457)   # from paper, typical value
    rs_truth = 10**0.4896        # from paper
    rstar_truth = 10**(-0.3363)  # from paper
    ra_truth = 10**(-0.4029)     # from paper

    rho_truth = np.log10(rho_func(r, rho0_truth, rs_truth, gamma_truth))
    mass_truth = np.log10(mass_func(r, rho0_truth, rs_truth, gamma_truth) + 1e-8)
    beta_truth = beta_func(r, rstar_truth, ra_truth)

    return rho_truth, mass_truth, beta_truth

truth_cored = get_truth(gamma_truth=0)
truth_cuspy = get_truth(gamma_truth=1)

# === Step 10: Plot ===
fig, axs = plt.subplots(3, 2, figsize=(12, 16))

titles = ['Cored profile γ=0', 'Cuspy profile γ=1']
summaries = [summary_cored, summary_cuspy]
truths = [truth_cored, truth_cuspy]

for col in range(2):
    rho_median, rho_low, rho_high, mass_median, mass_low, mass_high, beta_median, beta_low, beta_high = summaries[col]
    rho_truth, mass_truth, beta_truth = truths[col]

    # Density ρ(r)
    axs[0, col].plot(log_r, rho_median, 'k-', label='Posterior')
    axs[0, col].fill_between(log_r, rho_low, rho_high, color='blue', alpha=0.3)
    axs[0, col].plot(log_r, rho_truth, 'r--', label='Truth')
    axs[0, col].set_ylabel(r'$\log_{10} \rho\ (M_\odot/\mathrm{kpc}^3)$')

    # Mass M(r)
    axs[1, col].plot(log_r, mass_median, 'k-')
    axs[1, col].fill_between(log_r, mass_low, mass_high, color='blue', alpha=0.3)
    axs[1, col].plot(log_r, mass_truth, 'r--')
    axs[1, col].set_ylabel(r'$\log_{10} (M/M_\odot)$')

    # Beta β(r)
    axs[2, col].plot(log_r, beta_median, 'k-')
    axs[2, col].fill_between(log_r, beta_low, beta_high, color='blue', alpha=0.3)
    axs[2, col].plot(log_r, beta_truth, 'r--')
    axs[2, col].set_ylabel(r'$\beta(r)$')
    axs[2, col].set_xlabel(r'$\log_{10}(r/r_\star)$')

    axs[0, col].set_title(titles[col])
    axs[0, col].legend()

plt.tight_layout()
plt.savefig('Figures/figure2.png', dpi=300)
print("Saved Figure 2 as figure2.png")
plt.show()
