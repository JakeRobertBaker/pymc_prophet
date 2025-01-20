import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import nutpie

print(f"Running on PyMC v{pm.__version__}")

# Initialize random number generator
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# set the genuine model constants
sigma_true = 1
alpha_true = [1, 1.5, 2.5]

# number of fake data points
N = 200
X1 = np.random.randn(N)
X2 = np.random.randn(N) * 0.2

# simulate epsilon as N(0, sigma**2)
epsilon = rng.normal(size=N) * sigma_true
Y = alpha_true[0] + alpha_true[1] * X1 + alpha_true[2] * X2 + epsilon

basic_model = pm.Model()

with basic_model:
    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10, shape=3)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Expected value of outcome
    mu = alpha[0] + alpha[1] * X1 + alpha[2] * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)


compiled_model = nutpie.compile_pymc_model(basic_model)
