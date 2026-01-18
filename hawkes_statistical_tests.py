import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import kstest
from Thinning import thinning_hawkes
from hawkes_inference import estimate_hawkes_params_, compensator_expo_at_times_split, build_cumulation

def build_cumulation(comp_values, lambda_Tmax_values):
    """
    Put he compensator sequence to form a unified timeline.
    """
    cumulation = []
    shift = 0.0 

    for r in range(len(comp_values)):
        seq = comp_values[r]
        # Shift each sequence to make them contiguous
        shifted_seq = [x + shift for x in seq]
        cumulation.append(shifted_seq)

        # Update the shift using the total intensity integral of the current process
        shift += lambda_Tmax_values[r]

    return cumulation

def collect_ks_stats(M, p, mu, alpha, beta, T, n, type, theta_factor=0.9, level=0.05, peri=2.0):
    """
    Performs Monte Carlo simulations to collect Kolmogorov-Smirnov statistics and pvaleur
    """
    stats, pvals, ms = [], [], []
    a = 0  # Counter for rejections based on alpha level
    for _ in range(M):
        # Randomly select p processes out of n total simulated processes
        idx = np.random.choice(n, size=p, replace=False)
        idx_set = set(idx.tolist())
        arrays_p, flat_all = [], []

        # Simulate n independent Hawkes processes
        for i in range(n):
            ts = thinning_hawkes(mu, alpha, beta, T, kernel_type=type, peri=peri).simulate()
            flat_all.extend(ts)
            if i in idx_set:
                arrays_p.append(np.array(ts, dtype=float))

        # Sort all events to estimate global parameters
        flat_sorted = np.array(sorted(flat_all), dtype=float)
        mu_hat, alpha_hat, beta_hat = estimate_hawkes_params_(flat_sorted, T, n)
        print(f"Estimated: mu={mu_hat:.4f}, alpha={alpha_hat:.4f}, beta={beta_hat:.4f}")

        # Compute compensators for the selected subset of processes
        comp_values, lambda_Tmax_values = [], []
        for arr in arrays_p:
            comp_seq, comp_Tmax = compensator_expo_at_times_split(arr, mu_hat, alpha_hat, beta_hat, T)
            comp_values.append(comp_seq)
            lambda_Tmax_values.append(comp_Tmax)

        # Cumulative transformed time points
        cumulation = build_cumulation(comp_values, lambda_Tmax_values)
        theta = theta_factor * (np.sum(lambda_Tmax_values) / p)
        cutoff = p * theta

        # Flatten and filter transformed points within the cutoff range
        new = np.array([x for seq in cumulation for x in seq], dtype=float)
        Nc_cut = new[new <= cutoff]
        if Nc_cut.size == 0:
            continue
        U = Nc_cut / cutoff
        stat, pval = kstest(U, "uniform")
        stats.append(stat)
        
        # Track rejection rate
        if pval < level:
            a += 1
        ms.append(Nc_cut.size)

    # Return stats, number of rejection H_0, and sample sizes
    return np.array(stats), a / M, np.array(ms)

def run_and_save_experiment(param_name, param_values, fixed_params, kernels, M, p, n, T, alpha_sig=0.05):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"\n Figures will be saved in: {BASE_DIR}\n")

    for kernel in kernels:
        proportions = []
        
        print(f"\n>>> Running: Kernel={kernel}, Variable={param_name}")
        
        for val in param_values:

            current_params = fixed_params.copy()
            current_params[param_name] = val
            
            print(f" Testing {param_name} = {val}...", end=" ")
            
            _, rejection, _ = collect_ks_stats(
                M, p, 
                current_params['mu'], 
                current_params['alpha'], 
                current_params['beta'], 
                T, n, kernel, 
                theta_factor=0.9,
                level=alpha_sig, 
                peri=current_params['peri']
            )
            proportions.append(rejection)
            print(f"Done (Rejection: {rejection})")
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            param_values, proportions,
            marker='s', markersize=8,
            linestyle='-', label='Rejection Proportion'
        )
        
        plt.axhline(
            y=alpha_sig,
            linestyle='--',
            label=f'Alpha Level ({alpha_sig})'
        )
        
        plt.xlabel(f"Parameter Value: {param_name}", fontsize=12)
        plt.ylabel("Proportion of Rejecting H0", fontsize=12)
        plt.title(
            f"Sensitivity Analysis: {param_name} vs Rejection Rate\n"
            f"[Kernel: {kernel}, M={M}, n={n}]",
            fontsize=14
        )
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        filename = f"plot_{kernel}_{param_name}.png"
        full_path = os.path.join(BASE_DIR, filename)

        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved at: {full_path}")
