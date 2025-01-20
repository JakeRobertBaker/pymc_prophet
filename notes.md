# Setup
Conda
```
mamba create -c conda-forge -n pymc_env "pymc>=5" jupyter
```

### Additive and Multiplicative interaction

According to [forecaster.py](https://github.com/facebook/prophet/blob/101dd50e3195c1875ee856cdf49ee9fcd6a87fa3/python/fbprophet/forecaster.py#L1191-L1194) the regressors combine as 

$$
y = \text{trend} * (1 + \text{multiplicative terms}) + \text{additive terms}
$$