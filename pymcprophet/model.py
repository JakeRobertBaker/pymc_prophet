from typing import Literal
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import numpy as np

from pydantic import BaseModel, Field, field_validator


class BayesTSConfig(BaseModel):
    growth: Literal["linear", "logistic", "flat"] = "linear"
    y_scale_type: Literal["absmax", "minmax"] = "absmax"
    logistic_floor: bool = False
    daily_seasonality: Literal["enabled", "disabled", "auto"] = "auto"
    weekly_seasonality: Literal["enabled", "disabled", "auto"] = "auto"
    yearly_seasonality: Literal["enabled", "disabled", "auto"] = "auto"
    seasonality_prior_scale: float = Field(10.0, gt=0)
    seasonality_mode: Literal["additive", "multiplicative"] = "additive"

    @field_validator("logistic_floor")
    def check_logistic_floor(cls, v, values):
        if v and values.get("growth") != "logistic":
            raise ValueError("logistic_floor=True requires growth='logistic'")
        return v


class ModelCol(BaseModel):
    name: str
    mode: Literal["additive", "multiplicative"]


class SeasonalityTermConfig(ModelCol):
    period: float = Field(gt=0)
    fourier_order: int = Field(gt=0)


class BayesTS:
    def __init__(self, config: BayesTSConfig):
        self.config = config

        self.logisitc_terms = []
        self.regressors: list[ModelCol] = []
        self.seasonalities: list[SeasonalityTermConfig] = []

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

        if self.config.growth == "logisitc":
            self.logisitc_terms.append("cap")
            if self.config.logistic_floor:
                self.logisitc_terms.append("floor")

        self.regressors.extend([ModelCol(name=reg, mode="additive") for reg in additive_regressors])
        self.regressors.extend([ModelCol(name=reg, mode="multiplicative") for reg in multiplicative_regressors])

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
        # force through the manually turned on seasnalities
        if self.config.daily_seasonality == "enabled":
            self.add_daily_seasonality()

        if self.config.weekly_seasonality == "enabled":
            self.add_weekly_seasonality()

        if self.config.yearly_seasonality == "enabled":
            self.add_yearly_seasonality()

        range = self.raw_model_df["ds"].max() - self.raw_model_df["ds"].min()
        dt = self.raw_model_df["ds"].diff()
        min_dt = dt.iloc[dt.values.nonzero()[0]].min()

        # yearly if therea are >= 2 years of history
        if self.config.yearly_seasonality == "auto" and range >= pd.Timedelta(days=730):
            self.add_yearly_seasonality()

        # weekly if there are >= 2 weeks of history and there exists spacing < 7 days
        if self.config.weekly_seasonality == "auto" and (range >= pd.Timedelta(weeks=2) and min_dt < pd.Timedelta(weeks=1)):
            self.add_weekly_seasonality()

        # daily if there are >= 2 days of history and there exists spacing < 1 day
        if self.config.daily_seasonality == "auto" and (range >= pd.Timedelta(days=2) and min_dt < pd.Timedelta(days=1)):
            print(min_dt)
            self.add_daily_seasonality()

    def add_seasonality(self, name: str, period: float, fourier_order: int, mode: Literal["additive", "multiplicative"]):
        if name in [term.name for term in self.seasonalities]:
            raise ValueError(f"Seasonality term '{name}' already exists.")

        self.seasonalities.append(SeasonalityTermConfig(name=name, period=period, fourier_order=fourier_order, mode=mode))

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
        for term in self.seasonalities:
            n_vals = np.arange(1, term.fourier_order + 1)
            t = np.outer(n_vals, df["t_seasonality"] * 2 * np.pi / term.period)  # shape (fourier_order, T)

            sin_col_names, cos_col_names = self._get_seasonality_term_col_names(term)

            sin_terms = np.sin(t)  # shape (fourier_order, T)
            sin_df = pd.DataFrame(sin_terms.T, columns=sin_col_names)

            cos_terms = np.cos(t)  # shape (fourier_order, T)
            cos_df = pd.DataFrame(cos_terms.T, columns=cos_col_names)

            df = pd.concat([df, sin_df, cos_df], axis=1)

        return df

    def _get_seasonality_term_col_names(self, term: SeasonalityTermConfig):
        """
        Get the col names of the seasonality term
        """
        n_vals = np.arange(1, term.fourier_order + 1)
        sin_col_names = [f"{term.name}_sin_{n}" for n in n_vals]
        cos_col_names = [f"{term.name}_cos_{n}" for n in n_vals]
        return sin_col_names, cos_col_names

    def get_seasonality_term_col_names(self, term: SeasonalityTermConfig) -> list[str]:
        sin_col_names, cos_col_names = self._get_seasonality_term_col_names(term)
        col_names = sin_col_names + cos_col_names
        return col_names

    def get_all_seasonality_col_names(self):
        return [
            col_name
            for col_names in [self.get_seasonality_term_col_names(term) for term in self.seasonalities]
            for col_name in col_names
        ]

    def transform_y(self, df: pd.DataFrame):
        """
        Apply affine transform:
            y(t) ->  ( y(t) - shift(t) ) / scale
        """
        if not self.y_scales_set:
            raise ValueError("Need to assign data to the model to set transform parms.")

        if self.config.logistic_floor:
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
            if self.config.logistic_floor:
                self.scale = (self.raw_model_df["y"] - self.raw_model_df["floor"]).abs().max()
            else:
                self.scale = self.raw_model_df["y"].abs().max()

        if self.config.y_scale_type == "minmax":
            if self.config.logistic_floor:
                self.scale = self.raw_model_df["cap"].max() - self.raw_model_df["floor"].min()
            else:
                self.scale = self.raw_model_df["y"].max() - self.raw_model_df["y"].min()

        self.y_scales_set = True

    def get_all_model_cols(self, by_type=False):
        # there are input cols and processed seasonality cols
        cols_by_type = self.get_input_model_cols(by_type=True)
        cols_by_type.update({"seasonalities": self.get_all_seasonality_col_names()})

        if by_type:
            return cols_by_type
        else:
            return [col_name for col_list in cols_by_type.values() for col_name in col_list]

    def get_input_model_cols(self, by_type=False):
        cols_by_type = {"fundamental": ["ds", "y"]}
        cols_by_type.update({"logistic": self.logisitc_terms})
        cols_by_type.update({"regressors": [r.name for r in self.regressors]})

        if by_type:
            return cols_by_type
        else:
            return [col_name for col_list in cols_by_type.values() for col_name in col_list]
