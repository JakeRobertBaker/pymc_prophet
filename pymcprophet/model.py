from typing import Literal
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import numpy as np


from pymcprophet.model_templates import (
    BayesTSConfig,
    Feature,
    RegressorFeature,
    FeatureFamily,
    RegressorFamily,
    SeasonalityFamily,
    ModelSpecification,
)


class BayesTS:
    def __init__(self, config: BayesTSConfig):
        self.config = config
        self.model_spec = ModelSpecification()

        if config.growth == "logistic":
            logisitc_fam = FeatureFamily(features={"cap": Feature()})
            if config.floor_logistic:
                logisitc_fam.add_feature(feature_name="floor", feature=Feature())

            self.model_spec.add_feature_family("logistic", logisitc_fam)

        self.data_assigned = False
        self.y_scales_set = False

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
            ValueError("Data already assigned to model")

        additive_regressor_family = RegressorFamily(
            regressor_type="extra_regressor",
            mode="additive",
            features={reg: RegressorFeature() for reg in additive_regressors},
        )
        self.model_spec.add_regressor_family("additive_regressors", additive_regressor_family)

        multiplicative_regressor_family = RegressorFamily(
            regressor_type="extra_regressor",
            mode="additive",
            features={reg: RegressorFeature() for reg in additive_regressors},
        )
        self.model_spec.add_regressor_family("multiplicative_regressors", multiplicative_regressor_family)

        self.validate_input_matrix(df)

        self.raw_model_df = df[self.get_input_model_cols()]
        self.set_y_scale()
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
        if (self.config.yearly_seasonality == "auto" and range >= pd.Timedelta(days=730)) or (
            self.config.yearly_seasonality == "enabled"
        ):
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

    def add_seasonality(self, name: str, period: float, fourier_order: int, mode: Literal["additive", "multiplicative"]):
        # TODO move this validation in to the template
        if name in [s_fam_name for s_fam_name in self.model_spec.seasonality_families.keys()]:
            raise ValueError(f"Seasonality term '{name}' already exists.")

        self.model_spec.add_seasonal_family(name, SeasonalityFamily(mode=mode, period=period, fourier_order=fourier_order))

    def add_daily_seasonality(self, fourier_order: int = 4):
        self.add_seasonality("daily", period=1, fourier_order=fourier_order, mode=self.config.seasonality_mode)

    def add_weekly_seasonality(self, fourier_order: int = 3):
        self.add_seasonality("weekly", period=7, fourier_order=fourier_order, mode=self.config.seasonality_mode)

    def add_yearly_seasonality(self, fourier_order: int = 10):
        self.add_seasonality("yearly", period=365.25, fourier_order=fourier_order, mode=self.config.seasonality_mode)

    def validate_input_matrix(self, df: pd.DataFrame):
        """
        Validate that neccessary input cols and types are present.
        Derived cols such as seasonality are not required.
        """
        missing_cols = [
            f"missing {col_type}: {col} "
            for col_type, col_list in self.get_input_model_cols(by_type=True).items()
            for col in col_list
            if col not in df.columns
        ]
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

        # add seasonality cols
        for s_fam_name, term in self.model_spec.seasonality_families.items():
            # calculate terms
            n_vals = np.arange(1, term.fourier_order + 1)
            t = np.outer(n_vals, df["t_seasonality"] * 2 * np.pi / term.period)  # shape (fourier_order, T)
            sin_terms = np.sin(t)  # shape (fourier_order, T)
            cos_terms = np.cos(t)  # shape (fourier_order, T)

            # create names in the same order
            sin_col_names = [f"{s_fam_name}_sin_{n}" for n in n_vals]
            cos_col_names = [f"{s_fam_name}_cos_{n}" for n in n_vals]
            # vital we update term as we go
            for name in sin_col_names + cos_col_names:
                term.add_feature(name, RegressorFeature(feature_origin="generated"))

            sin_df = pd.DataFrame(sin_terms.T, columns=sin_col_names)
            cos_df = pd.DataFrame(cos_terms.T, columns=cos_col_names)
            df = pd.concat([df, sin_df, cos_df], axis=1)

        return df

    def get_seasonality_term_col_names(self, term: SeasonalityFamily):
        return term.get_cols()

    def get_all_seasonality_col_names(self):
        return self.model_spec.get_seasonality_cols()

    def transform_y(self, df: pd.DataFrame):
        """
        Apply affine transform:
            y(t) ->  ( y(t) - shift(t) ) / scale
        """
        if not self.y_scales_set:
            raise ValueError("Need to assign data to the model to set transform parms.")

        if self.config.floor_logistic:
            shift = df["floor"]
        else:
            shift = 0 if self.config.y_scale_type == "absmax" else self.raw_model_df["y"].min()

        df["y_trans"] = (df["y"] - shift) / self.scale

        return df

    def set_y_scale(self):
        """
        Set the scale term in transform:
            y(t) ->  ( y(t) - shift(t) ) / scale
        """

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

    def get_model_cols(self, by_type=False, input_only=False):
        # there are input cols and generated cols
        cols_by_type = {"fundamental": ["ds", "y"]}

        cols_by_type.update({"logistic": self.model_spec.get_feature_cols()})
        cols_by_type.update({"regressors": self.model_spec.get_regressor_cols()})

        # columns that are not required as model input, they are generated
        if not input_only:
            cols_by_type.update({"seasonalities": self.model_spec.get_seasonality_cols()})

        if by_type:
            return cols_by_type
        else:
            return [col_name for col_list in cols_by_type.values() for col_name in col_list]

    def get_input_model_cols(self, by_type=False):
        return self.get_model_cols(by_type=by_type, input_only=True)
