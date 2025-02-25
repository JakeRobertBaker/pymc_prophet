import pymc as pm
import numpy as np

import pytensor
pytensor.config.cxx = "/usr/bin/clang++"



# Generate some data
np.random.seed(123)
observed_data = np.random.normal(loc=0, scale=1, size=100)

# Define the model
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=observed_data)
    
    # Perform posterior sampling
    trace = pm.sample(1000, return_inferencedata=False)

# Print summary of the trace
# print(pm.summary(trace))