import numpy as np
import star_sampler as ss   # star_sampler.py in working directory
from template import Model  # template.py in working directory
from multiprocessing import Pool, cpu_count

# Number of processes
N_PROCESSES = 30  # adjust as needed, up to cpu_count()

# Sampling function for a single galaxy
def sample_one(_):
    # Priors (App. A.4)
    rho0   = 10**np.random.uniform(5,   8)
    rs     = 10**np.random.uniform(-1,  np.log10(5))
    gamma  = np.random.uniform(-1,     2)
    r_star = np.random.uniform(0.2*rs, 1*rs)
    r_a    = np.random.uniform(0.5*r_star, 2*r_star)
    params = dict(rho0=rho0, rs=rs, gamma=gamma,
                  r_star=r_star, r_a=r_a)
    # Generate kinematics
    R, vlos, _ = generate_galaxy_sample(params)
    return R, vlos, params

# Wrap existing generate_galaxy_sample for multiprocessing

def generate_galaxy_sample(params, mu_stars=100, vel_error=0.1,
                           steps=50, resample_factor=10):
    # 1) draw number of tracer stars
    n_stars = np.random.poisson(mu_stars)
    # 2) initialize the DF model
    model = Model(**params)
    # 3) importance-sample (r, vr, vt)
    arr = ss.impt_sample(
        model_class=model,
        steps=steps,
        resample_factor=resample_factor,
        samplesize=n_stars,
        replace=True,
        r_vr_vt=False
    )
    # unpack
    r, vr, vt = arr[0], arr[1], arr[2]
    # 4) convert to Cartesian
    x, y, z, vx, vy, vz = ss.r_vr_vt_complete(np.vstack((r, vr, vt)).T)
    # 5) random projection
    axis = np.random.choice(['x','y','z'])
    if axis=='x':
        R = np.sqrt(y**2 + z**2); vlos=vx
    elif axis=='y':
        R = np.sqrt(x**2 + z**2); vlos=vy
    else:
        R = np.sqrt(x**2 + y**2); vlos=vz
    # 6) add noise
    vlos_noisy = vlos + np.random.normal(0, vel_error, size=vlos.shape)
    return R, vlos_noisy, params

# Parallel dataset generation

def generate_split(n_samples):
    # Use Pool to parallelize sample_one over n_samples
    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(sample_one, range(n_samples))
    # Unpack
    data   = [(res[0], res[1]) for res in results]
    labels = [res[2] for res in results]
    return data, labels

if __name__ == '__main__':
    # Generate splits
    print(f"Generating train split ({40000} samples) on {N_PROCESSES} cores...")
    train_data, train_labels = generate_split(40000)
    print("Train split done.")

    print(f"Generating val split ({5000} samples)...")
    val_data, val_labels = generate_split(5000)
    print("Val split done.")

    print(f"Generating test split ({5000} samples)...")
    test_data, test_labels = generate_split(5000)
    print("Test split done.")

    # Save
    np.save('train_data.npy', train_data)
    np.save('train_labels.npy', train_labels)
    np.save('val_data.npy',   val_data)
    np.save('val_labels.npy', val_labels)
    np.save('test_data.npy',  test_data)
    np.save('test_labels.npy', test_labels)

    print("All splits generated and saved!")
