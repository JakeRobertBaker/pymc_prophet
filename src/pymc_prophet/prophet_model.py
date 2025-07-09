import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymc as pm
from arviz import InferenceData

from pymc_prophet.bayes_model import BayesTS
from pymc_prophet.model_config import BayesTSConfig


class ProphetModel(BayesTS):
    """
    The Bayes TS class with a defined model, fit and predict methods.
    """

    def __init__(self, config: BayesTSConfig):
        super().__init__(config)

    def predict(self, future_df: pd.DataFrame) -> InferenceData:
        future_model_df = self.produce_model_matrix(future_df)
        coords = {"time": future_model_df["ds"].to_numpy()}
        t_np, X_additive, X_multiplicative = self._produce_modelling_data(future_model_df)

        with self.forecast_model:
            new_data = {"t": t_np}
            if self.additive_feature_dict:
                new_data["additive_predictors"] = X_additive
                coords["additive_features"] = list(self.additive_feature_dict.keys())
            if self.multiplicative_feature_dict:
                new_data["multiplicative_predictors"] = X_multiplicative
                coords["multiplicative_features"] = list(self.multiplicative_feature_dict.keys())

            pm.set_data(new_data, coords=coords)
            out_sample_predictive = pm.sample_posterior_predictive(self.posterior, predictions=True)

            return out_sample_predictive

    def _fit(self, raw_df: pd.DataFrame, **kwargs):
        self._produce_pymc_model()

        sample_args = {"chains": 4, "draws": 2000}
        sample_args.update(kwargs)

        with self.forecast_model:
            # sample posterior variables and deterministics, observed is not sampled
            self.posterior: InferenceData = pm.sample(**sample_args)
            # sample posterior predictive y_obs
            self.in_sample_predictive: InferenceData = pm.sample_posterior_predictive(self.posterior)

    def _produce_pymc_model(self):
        coords = {"time": self.model_df["ds"].to_numpy()}
        y = self.model_df["y_trans"].to_numpy()
        t_np, X_additive, X_multiplicative = self._produce_modelling_data(self.model_df)

        if self.additive_feature_dict:
            additive_prior_mus = np.array([feature.prior_params["mu"] for feature in self.additive_feature_dict.values()])
            additive_prior_sigmas = np.array([feature.prior_params["sigma"] for feature in self.additive_feature_dict.values()])
            coords["additive_features"] = list(self.additive_feature_dict.keys())

        if self.multiplicative_feature_dict:
            multiplicative_prior_mus = np.array(
                [feature.prior_params["mu"] for feature in self.multiplicative_feature_dict.values()]
            )
            multiplicative_prior_sigmas = np.array(
                [feature.prior_params["sigma"] for feature in self.multiplicative_feature_dict.values()]
            )
            coords["multiplicative_features"] = list(self.multiplicative_feature_dict.keys())

        forecast_model = pm.Model(coords=coords)

        with forecast_model:
            # trend(t) applies to k(t-m),
            k = pm.Normal("k", mu=0, sigma=5)
            m = pm.Normal("m", mu=0, sigma=5)
            t = pm.Data("t", t_np, dims="time")
            trend = pm.Deterministic("trend", k * t + m, dims="time")

            # defaults get overwritten by additive or multiplicative terms if present
            additive_terms = 0
            multiplicative_terms = 0

            if self.additive_feature_dict:
                additive_predictors = pm.Data("additive_predictors", X_additive, dims=["time", "additive_features"])
                beta_additive = pm.Normal(
                    "beta_additive",
                    mu=additive_prior_mus,
                    sigma=additive_prior_sigmas,
                    dims="additive_features",
                )
                additive_terms = pm.Deterministic(
                    "additive_terms",
                    additive_predictors @ beta_additive,
                    dims="time",
                )

            if self.multiplicative_feature_dict:
                multiplicative_predictors = pm.Data(
                    "multiplicative_predictors", X_multiplicative, dims=["time", "multiplicative_features"]
                )
                beta_multiplicative = pm.Normal(
                    "beta_multiplicative",
                    mu=multiplicative_prior_mus,
                    sigma=multiplicative_prior_sigmas,
                    dims="multiplicative_features",
                )
                multiplicative_terms = pm.Deterministic(
                    "multiplicative_terms",
                    multiplicative_predictors @ beta_multiplicative,
                    dims="time",
                )

            # predictions
            y_pred = pm.Deterministic("y_pred", trend * (1 + multiplicative_terms) + additive_terms, dims="time")

            # observation
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)
            y_obs = pm.Normal("y_obs", mu=y_pred, sigma=sigma_obs, observed=y, dims="time")  # noqa: F841

        self.forecast_model = forecast_model

    def plot(
        self, out_sample_predictive: InferenceData | None = None, out_sample_y_obs=None, sig=0.05, region_color="mediumblue"
    ):
        if out_sample_predictive:
            # plot out of sample predicitve
            pass
        if out_sample_y_obs:
            pass
            # plot out of sample y obs

        # in sample plots
        fig = go.Figure()

        # predictive
        fig.add_trace(
            go.Scatter(
                x=self.posterior.constant_data["time"],
                y=self.in_sample_predictive.posterior_predictive["y_obs"].quantile(q=1 - sig / 2, dim=("chain", "draw")),
                fill=None,
                mode="lines",
                line=dict(width=0, color=region_color),
                name=f"{100 - 50 * sig} percent",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.posterior.constant_data["time"],
                y=self.in_sample_predictive.posterior_predictive["y_obs"].quantile(q=sig / 2, dim=("chain", "draw")),
                fill="tonexty",  # fill area between trace0 and trace1
                mode="lines",
                line=dict(width=0, color=region_color),
                name=f"{50 * sig} percent",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.posterior.constant_data["time"],
                y=self.in_sample_predictive.posterior_predictive["y_obs"].mean(("chain", "draw")),
                name="y_pred_mean",
                mode="lines",
                line=dict(width=1, color=region_color),
            )
        )

        # actuals
        fig.add_trace(
            go.Scatter(
                x=self.posterior.constant_data["time"],
                y=self.posterior.observed_data["y_obs"],
                mode="markers",
                marker=dict(color="red", size=4, opacity=0.6),
                name="y actual",
            )
        )

        return fig
