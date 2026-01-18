import numpy as np
import math
from scipy.optimize import minimize

def estimate_hawkes_params_(timestamps, T, n):
    """
    Estimates Hawkes process parameters (mu, alpha, beta) using Maximum Likelihood Estimation under H_0
    """
    def neg_log_likelihood(params):
        mu, alpha, beta = params
        #  mu and beta must be positive, alpha must be non-negative
        if mu <= 0 or alpha < 0 or beta <= 0:
            return np.inf

        I, t_prev, ll_sum = 0.0, 0.0, 0.0
        # Recursive calculation of the log-likelihood sum
        for ti in timestamps:
            I *= math.exp(-beta * (ti - t_prev))
            lam = mu * n + I
            if lam <= 0:
                return np.inf
            ll_sum += math.log(lam)
            I += alpha  
            t_prev = ti

        # Calculate the compensator (integral of the intensity function over [0, T] by using n process)
        arr = np.array(timestamps)
        compensator = n * mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - arr)))

        # Return negative log-likelihood (That what we need to minimize)
        return -(ll_sum - compensator)

    # Optimization process using L-BFGS-B method
    res = minimize(
        fun=neg_log_likelihood,
        x0=[0.1, 0.1, 0.1],
        bounds=[(1e-8, None)] * 3,
        method="L-BFGS-B"
    )
    mu_hat, alpha_hat, beta_hat = res.x
    return mu_hat, alpha_hat, beta_hat

def compensator_expo_at_times_split(timestamps, mu, alpha, beta, T_max):
    """
    Calculates the compensator values at each event time and at the final time T_max under H_0
    """
    timestamps = np.array(timestamps)
    comp_seq = []
    
    for i, t in enumerate(timestamps):
        prev = timestamps[:i]    
        if i > 0:
            # Sum of the integrated kernels for all past events
            term = np.sum(1 - np.exp(-beta * (t - prev)))
        else:
            term = 0
        comp = mu * t + (alpha / beta) * term
        comp_seq.append(comp)

    # Final compensator value at Tmax
    if len(timestamps) > 0:
        term_Tmax = np.sum(1 - np.exp(-beta * (T_max - timestamps)))
    else:
        term_Tmax = 0

    comp_Tmax = mu * T_max + (alpha / beta) * term_Tmax

    return comp_seq, comp_Tmax

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