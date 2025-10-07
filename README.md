

# Recreating Dark Matter Density Distribution Figures Using Simple 2 Layer GCN

This repository contains all code and data needed to reproduce the main figures in **GNN.pdf**, which presents a graph-neural-network approach to inferring astrophysical parameters from simulated galaxy star-particle data.

---

## Repository Structure

```
├── StarSampler/  
│   ├── generate_stars.py       # Script to simulate star-particle distributions  
│   ├── utils.py                # Helper functions for data generation  
│   └── …                        
├── SimpleGCN.py                # Train a 2-layer GCN on the star data  
├── PlotFigure1.py              # Generate Figure 1: prediction vs truth & residuals  
├── PlotFigure2.py              # Generate Figure 2: density, mass, anisotropy posteriors  
├── PlotFigure3.ipynb           # Notebook to generate the overlaid corner plots  
├── data/                       # Example data (already generated)  
│   ├── training.h5             # HDF5 of 80,000 simulated galaxies  
│   ├── test.h5                 # HDF5 test set  
│   ├── ground_truths.csv       # True parameters for test set  
│   ├── predictions.csv         # GCN predictions on test set  
│   └── …                        
└── README.md                   # ← You are here  
```

---

## Pipeline Overview

1. **(Optional)** Generate star-particle data with the `StarSampler/` folder  
2. Train a basic Graph Convolutional Network (GCN) using `SimpleGCN.py`  
3. Produce Figures 1–3 with the plotting scripts and notebook  

> **Note:** The `data/` folder already contains example inputs and outputs. You do **not** need to re-run the expensive data generation or training steps unless you want to reproduce them from scratch.

---

## Step 1). Data Generation: `StarSampler/`

The **StarSampler** module simulates realistic star-particle distributions for a large ensemble of galaxies:

- **`generate_stars.py`**  
  - Uses astrophysical profiles (cored vs. cuspy halos) to sample 3D star positions.  
  - Outputs an HDF5 file (`features`, `labels`, and `ptr` arrays) that can be directly consumed by `SimpleGCN.py`.  
- **Workflow**  
  1. Define halo parameters (γ, rₛ, ρ₀, r\*, rₐ) for each simulated galaxy.  
  2. Sample *N* stellar positions from the 3D density profile.  
  3. Store features (`[x, y, z]`), labels (`[parameters]`), and pointer array (`ptr`) to separate graphs.

> **Performance:** Generating 80,000 galaxies without parallelization can take on the order of **1 month**. Pre-generated data is included in `data/training.h5`.

---

## Step 2). Training the GCN: `SimpleGCN.py`

This script implements and trains a 2-layer Graph Convolutional Network using **PyTorch Geometric**:

- **Key components**  
  - **`StarSamplerGraphDataset`**: loads `data/training.h5`, builds k-NN graphs on node positions.  
  - **`StellarGNN`**: two `GCNConv` layers + global mean pool + final linear layer to predict 5 parameters.  
- **Usage**  
  ```bash
  python SimpleGCN.py
  ```
  - Splits the dataset into 80% train / 20% validation  
  - Trains with early stopping (patience = 10 epochs)  
  - Saves best model to `best_model.pt`  
  - Evaluates on `data/test.h5`, writes:  
    - `data/predictions.csv` (model outputs)  
    - `data/ground_truths.csv` (true labels)  

---

## Step 3). Figure Generation

All three plotting scripts assume you have `predictions.csv` and `ground_truths.csv` in `data/`.

### 3.1 `PlotFigure1.py`  
Recreates **Figure 1** of GNN.pdf:  
- **Top row**: binned “predicted vs. true” scatter with median line, 68% & 95% credible bands, and 1:1 reference line.  
- **Bottom row**: residuals (`prediction − truth`) vs. truth for γ, log₁₀ rₛ, and log₁₀ ρ₀.  
- **Output**: `Figures/figure1.png`

### 3.2 `PlotFigure2.py`  
Recreates **Figure 2**:  
- Computes density ρ(r), enclosed mass M(r), and anisotropy β(r) profiles for:  
  - **Cored** posterior samples (γ = 0)  
  - **Cuspy** posterior samples (γ = 1)  
- Plots median curves ±16/84% credible regions, overlaid with ground-truth curves.  
- **Output**: `Figures/figure2.png`

### 3.3 `PlotFigure3.ipynb`  
Recreates **Figure 3**:  
- Loads bootstrap-resampled posterior draws for two example galaxies (indices 260 & 59).  
- Uses the `corner` package to overlay red & blue contour plots of (γ, rₛ, ρ₀), showing 68% & 95% credible regions and truth lines.  
- **Output**: Jupyter notebook figure (can be exported to `Figures/figure3.png`).

---

## Dependencies

- Python 3.8+  
- PyTorch 1.12+ & PyTorch Geometric  
- NumPy, pandas, h5py, scikit-learn, matplotlib, corner, scipy  

Install via:

```bash
pip install torch torch-geometric numpy pandas h5py scikit-learn matplotlib corner scipy
```

---

## Quick Start

1. **Use provided data**  
   ```bash
   ls data/
   # training.h5, test.h5, ground_truths.csv, predictions.csv
   ```
2. **Train the model** (optional, uses ~1–2 hours on GPU)  
   ```bash
   python SimpleGCN.py
   ```
3. **Generate Figures**  
   ```bash
   python PlotFigure1.py
   python PlotFigure2.py
   jupyter nbconvert --to notebook --execute PlotFigure3.ipynb
   ```
4. **View outputs** in `Figures/` and embed into your README or paper.

---

## Notes

- To **re-generate** the star data from scratch:  
  ```bash
  cd StarSampler
  python generate_stars.py --n_galaxies 80000
  ```
- For faster data generation, consider parallelization or cluster computing.
- The StarSampler program is empty (as in, you have to fill in the blanks). This is done so that you can fine tune the sampling process for your needs.
- Feel free to tune hyperparameters (learning rate, batch size, network depth) in `SimpleGCN.py` to explore performance.
