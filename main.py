import numpy as np
from hawkes_statistical_tests import run_and_save_experiment

M = 3
n = 30
p = int((n)**(2/3))
T = 20

# Baseline parameters
base_params = {'mu': 0.3, 'alpha': 0.5, 'beta': 1, 'peri': 2}
kernels = [ 'exponential', 'exponential_periodic', 'gaussien','box_periodic',  'inhibitory_exponential','box_decale']

# Define the ranges for your 3 experiments
beta_range = [0.6, 0.8, 1, 1.5, 1.8]
mu_range = [0.3, 0.8 ,1, 1.5]
peri_range = [0.5, 1, 1.5, 2.0 ,5.0 ]
# 1. Varying Beta

run_and_save_experiment('beta', beta_range, base_params, kernels, M, p, n, T)

# 2. Varying Mu 
run_and_save_experiment('mu', mu_range, base_params, kernels, M, p, n, T)

# 3. Varying Periodic (peri)
run_and_save_experiment('peri', peri_range, base_params, kernels, M, p, n, T)