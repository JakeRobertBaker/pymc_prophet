from typing import Literal
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import numpy as np

from pymcprophet.model_templates import BayesTSConfig, Feature, RegressorFeature, ModelSpecification, SeasonalityFeature


class BayesTS:
    def __init__(self, config: BayesTSConfig):
        self.config = config
        self.model_spec = ModelSpecification()

        if config.growth == "logistic":
            self.model_spec.add_feature("cap", Feature(family_name="logistic"))
            if config.floor_logistic:
                self.model_spec.add_feature("floor", Feature(family_name="logistic"))

        self.data_assigned = False
        self.y_scales_set = False
        self.seasonality_families = {}

    def add_regressor(self):
        pass

    def assign_model_matrix(
        self,
        df: pd.DataFrame,
        additive_regressors: list[str] = [],
        multiplicative_regressors: list[str] = [],
    ):
        """
        Assign the model data to model class instance.
        This is done once per model.
        Assigns things like scale params.
        """
        if self.data_assigned:
            raise ValueError("Data already assigned to model")

        # TODO rarefactor to add regressor of desired prior

        regressor_prior_params = {"mu": 0, "sigma": self.config.regressor_prior_scale}

        for reg_name in additive_regressors:
            self.model_spec.add_feature(
                reg_name,
                RegressorFeature(
                    family_name="additive_regressors", mode="multiplicative", prior_kind="normal", prior_params=regressor_prior_params
                ),
            )

        for reg_name in multiplicative_regressors:
            self.model_spec.add_feature(
                reg_name,
                RegressorFeature(
                    family_name="multiplicative_regressors", mode="multiplicative", prior_kind="normal", prior_params=regressor_prior_params
                ),
            )

        self.validate_input_matrix(df)

        self.raw_model_df = df[self.get_input_model_cols()]
        self._set_y_scale()
        self.determine_seasonalities()

        # use class information to produce the model matrix
        self.model_df = self._produce_model_matrix(df)
        self.data_assigned = True

    def determine_seasonalities(self):
        """
        Of the seasonalities with config set to auto, determine which are eligible.
        Mimics the same criteria that Prophet uses.
        """

        range = self.raw_model_df["ds"].max() - self.raw_model_df["ds"].min()
        dt = self.raw_model_df["ds"].diff()
        min_dt = dt.iloc[dt.values.nonzero()[0]].min()

        # yearly if therea are >= 2 years of history
        if (self.config.yearly_seasonality == "auto" and range >= pd.Timedelta(days=730)) or (self.config.yearly_seasonality == "enabled"):
            self.add_yearly_seasonality()

        # weekly if there are >= 2 weeks of history and there exists spacing < 7 days
        if (self.config.weekly_seasonality == "auto" and range >= pd.Timedelta(weeks=2) and min_dt < pd.Timedelta(weeks=1)) or (
            self.config.weekly_seasonality == "enabled"
        ):
            self.add_weekly_seasonality()

        # daily if there are >= 2 days of history and there exists spacing < 1 day
        if (self.config.daily_seasonality == "auto" and range >= pd.Timedelta(days=2) and min_dt < pd.Timedelta(days=1)) or (
            self.config.daily_seasonality == "enabled"
        ):
            self.add_daily_seasonality()

    def add_seasonality_family(self, name: str, period: float, fourier_order: int, mode: Literal["additive", "multiplicative"]):
        if self.data_assigned:
            raise ValueError("We do not support adding seasonalities after data is assigned. Do that before.")

        if name in self.seasonality_families:
            raise ValueError("A seasonality of the same name has already been added.")

        self.seasonality_families[name] = {"peroid": period, "fourier_order": fourier_order, "mode": mode}

    def add_daily_seasonality(self, fourier_order: int = 4):
        self.add_seasonality_family("daily", period=1, fourier_order=fourier_order, mode=self.config.seasonality_mode)

    def add_weekly_seasonality(self, fourier_order: int = 3):
        self.add_seasonality_family("weekly", period=7, fourier_order=fourier_order, mode=self.config.seasonality_mode)

    def add_yearly_seasonality(self, fourier_order: int = 10):
        self.add_seasonality_family("yearly", period=365.25, fourier_order=fourier_order, mode=self.config.seasonality_mode)

    def validate_input_matrix(self, df: pd.DataFrame):
        """
        Validate that neccessary input cols and types are present.
        Derived cols such as seasonality are not required.
        """
        missing_cols = []
        if "y" not in df.columns:
            missing_cols.append("y the indep var")
        if "ds" not in df.columns:
            missing_cols.append("ds the time var")

        for feature_name, feature in self.model_spec.get_input_feature_dict().items():
            if feature_name not in df.columns:
                missing_cols.append(f"col {feature_name} of family {feature.family_name} of type {feature.family_type}")

        if missing_cols:
            raise ValueError(f"Required cols missing,\n\t{', '.join(missing_cols)}")

        if not is_datetime64_any_dtype(df["ds"]):
            raise ValueError(f"Col 'ds' dtype {df['ds'].dtype} is not datetime")

    def produce_model_matrix(self, df: pd.DataFrame):
        """
        This wrapped version of produce applies to future dataframes too. Need to validate.
        """
        self.validate_input_matrix(df)
        return self._produce_model_matrix(df)

    def _produce_model_matrix(self, df: pd.DataFrame):
        """
        Used in assign model matrix but made generic so we can apply to future df.
        """

        # scale y var
        df = self.transform_y(df)

        # seasonality uses days since epoch
        day_to_nanosec = 3600 * 24 * int(1e9)
        dates = df["ds"].to_numpy(dtype=np.int64) / day_to_nanosec
        df["t_seasonality"] = dates

        for family_name, family_dict in self.seasonality_families.items():
            # calculate terms
            n_vals = np.arange(1, family_dict["fourier_order"] + 1)
            t = np.outer(n_vals, df["t_seasonality"] * 2 * np.pi / family_dict["peroid"])
            sin_terms = np.sin(t)  # shape (fourier_order, T)
            cos_terms = np.cos(t)  # shape (fourier_order, T)

            # create names in the same order
            sin_col_names = [f"{family_name}_sin_{n}" for n in n_vals]
            cos_col_names = [f"{family_name}_cos_{n}" for n in n_vals]

            # When we do this for the first time ass these created cols to the model spec
            if not self.data_assigned:
                for name in sin_col_names + cos_col_names:
                    # {"peroid": period, "fourier_order": fourier_order, "mode": mode}
                    self.model_spec.add_feature(
                        name,
                        SeasonalityFeature(
                            family_name=family_name,
                            period=family_dict["peroid"],
                            fourier_order=family_dict["fourier_order"],
                            mode=family_dict["mode"],
                            prior_kind="normal",
                            prior_params={"mu": 0, "sigma": self.config.seasonality_prior_scale},
                        ),
                    )

            sin_df = pd.DataFrame(sin_terms.T, columns=sin_col_names)
            cos_df = pd.DataFrame(cos_terms.T, columns=cos_col_names)
            df = pd.concat([df, sin_df, cos_df], axis=1)

        return df

    def transform_y(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply affine transform:
            y(t) ->  ( y(t) - shift(t) ) / scale
        """
        if not self.y_scales_set:
            raise ValueError("Need to assign data to the model to set transform parms.")

        df = input_df.copy()

        if self.config.floor_logistic:
            shift = df["floor"]
        else:
            shift = 0 if self.config.y_scale_type == "absmax" else self.raw_model_df["y"].min()

        df["y_trans"] = (df["y"] - shift) / self.scale

        return df

    def _set_y_scale(self):
        """
        Set the scale term in transform:
            y(t) ->  ( y(t) - shift(t) ) / scale
        """
        if self.y_scales_set:
            raise ValueError("y scale already set")

        if self.config.y_scale_type == "absmax":
            if self.config.floor_logistic:
                self.scale = (self.raw_model_df["y"] - self.raw_model_df["floor"]).abs().max()
            else:
                self.scale = self.raw_model_df["y"].abs().max()

        if self.config.y_scale_type == "minmax":
            if self.config.floor_logistic:
                self.scale = self.raw_model_df["cap"].max() - self.raw_model_df["floor"].min()
            else:
                self.scale = self.raw_model_df["y"].max() - self.raw_model_df["y"].min()

        self.y_scales_set = True

    def get_input_model_cols(self) -> list[str]:
        """
        Generate a list of column names required in input matrix.
        """
        return ["ds", "y"] + self.model_spec.get_input_feature_cols()
