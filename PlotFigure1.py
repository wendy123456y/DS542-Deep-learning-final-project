"""

This script performs the following major steps:
1. Load saved prediction and ground-truth arrays from CSV files.
2. Unpack the three parameters of interest (γ, log₁₀ rₛ, log₁₀ ρ₀).
3. Set up a 2×3 matplotlib grid for plotting summaries and residuals.
4. In each column, bin true values, compute median and credible intervals
   of predictions within each bin, and plot:
     - Top row: predicted vs. true with 68% & 95% CI bands and 1:1 line.
     - Bottom row: residuals (prediction minus truth) vs. true.
5. Save the resulting figure to disk and print a confirmation message.
"""

import numpy as np
import matplotlib.pyplot as plt

# === Step 1: Load your saved arrays (skip the header row) ===
preds  = np.loadtxt("data/predictions.csv", delimiter=",", skiprows=1)
truths = np.loadtxt("data/ground_truths.csv", delimiter=",", skiprows=1)

# === Step 2: Unpack into the three parameters we want ===
#    CSV columns are: [rho0, rs, gamma, r_star, r_a]
true_gamma    = truths[:, 1]
pred_gamma    = preds[:, 1]

true_log_rs   = truths[:, 0]
pred_log_rs   = preds[:, 0]

true_log_rho0 = truths[:, 2]
pred_log_rho0 = preds[:, 2]

# === Step 3: Set up a 2×3 grid for plotting ===
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
params = [
    (r"$\gamma$",            true_gamma,    pred_gamma),
    (r"$\log_{10}(r_s)$",    true_log_rs,   pred_log_rs),
    (r"$\log_{10}(\rho_0)$", true_log_rho0, pred_log_rho0),
]

nbins = 15
for col, (label, tvals, pvals) in enumerate(params):
    # === Step 4a: Define bins and compute summary statistics ===
    bins    = np.linspace(tvals.min(), tvals.max(), nbins+1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    med, p16, p84, p025, p975 = [], [], [], [], []
    for i in range(nbins):
        m = (tvals >= bins[i]) & (tvals < bins[i+1])
        if m.any():
            med.append(np.median(pvals[m]))
            p16.append(np.percentile(pvals[m], 16))
            p84.append(np.percentile(pvals[m], 84))
            p025.append(np.percentile(pvals[m], 2.5))
            p975.append(np.percentile(pvals[m], 97.5))
        else:
            med.append(np.nan); p16.append(np.nan)
            p84.append(np.nan); p025.append(np.nan)
            p975.append(np.nan)

    # === Step 4b: Top row – predicted vs. true with CI and 1:1 line ===
    ax = axes[0, col]
    ax.plot(centers, med,  color='blue', label='Median')
    ax.fill_between(centers, p16, p84,   color='blue', alpha=0.4, label='68% CI')
    ax.fill_between(centers, p025, p975, color='blue', alpha=0.2, label='95% CI')
    ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], 'r--', label='1:1')
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.legend()

    # === Step 4c: Bottom row – residuals vs. true ===
    ax = axes[1, col]
    ax.scatter(tvals, pvals - tvals, s=5, alpha=0.5)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(r"$\Delta\theta = \hat\theta - \theta_{\rm true}$")

# === Step 5: Save figure and print confirmation ===
plt.tight_layout()
plt.savefig("Figures/figure1.png", dpi=300)
print("Saved Figure 1 as figure1.png")
