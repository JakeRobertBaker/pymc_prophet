import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def _produce_numpyro_model(self):
    additive_feature_dict = self.additive_feature_dict
    multiplicative_feature_dict = self.multiplicative_feature_dict

    y = jnp.array(self.model_df["y_trans"].to_numpy())
    t_np = jnp.array(self.model_df["t"].to_numpy())
    X_additive = jnp.array(self.model_df[list(additive_feature_dict.keys())].to_numpy()) if additive_feature_dict else None
    X_multiplicative = jnp.array(self.model_df[list(multiplicative_feature_dict.keys())].to_numpy()) if multiplicative_feature_dict else None

    additive_prior_mus = jnp.array([feature.prior_params["mu"] for feature in additive_feature_dict.values()]) if additive_feature_dict else None
    additive_prior_sigmas = jnp.array([feature.prior_params["sigma"] for feature in additive_feature_dict.values()]) if additive_feature_dict else None
    multiplicative_prior_mus = jnp.array([feature.prior_params["mu"] for feature in multiplicative_feature_dict.values()]) if multiplicative_feature_dict else None
    multiplicative_prior_sigmas = jnp.array([feature.prior_params["sigma"] for feature in multiplicative_feature_dict.values()]) if multiplicative_feature_dict else None

    def model():
        k = numpyro.sample("k", dist.Normal(0, 5))
        m = numpyro.sample("m", dist.Normal(0, 5))
        trend = k * t_np + m

        additive_terms = 0.0
        if additive_feature_dict:
            beta_additive = numpyro.sample("beta_additive", dist.Normal(additive_prior_mus, additive_prior_sigmas))
            additive_terms = jnp.dot(X_additive, beta_additive)

        multiplicative_terms = 0.0
        if multiplicative_feature_dict:
            beta_multiplicative = numpyro.sample("beta_multiplicative", dist.Normal(multiplicative_prior_mus, multiplicative_prior_sigmas))
            multiplicative_terms = jnp.dot(X_multiplicative, beta_multiplicative)

        y_pred = trend * (1 + multiplicative_terms) + additive_terms
        sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(0.5))
        numpyro.sample("y_obs", dist.Normal(y_pred, sigma_obs), obs=y)

    self.numpyro_model = model