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


class SeasonalityTermConfig(BaseModel):
    name: str
    peroid: float = Field(gt=0)
    fourier_order: int = Field(gt=0)
    mode: Literal["additive", "multiplicative"] = "additive"


class BayesTS:
    def __init__(self, config: BayesTSConfig):
        self.config = config
        self.seasonalities = []
        self.data_assigned = False
        self.y_scales_set = False

        if self.config.daily_seasonality == "enabled":
            self.add_daily_seasonality()

        if self.config.weekly_seasonality == "enabled":
            self.add_weekly_seasonality()

        if self.config.yearly_seasonality == "enabled":
            self.add_yearly_seasonality()

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

        self.additive_regressors = additive_regressors
        self.multiplicative_regressors = multiplicative_regressors

        self.validate_matrix(df)

        raw_model_cols = ["ds", "y"]
        if self.config.growth == "logisitc":
            raw_model_cols += "cap"
            if self.config.logistic_floor:
                raw_model_cols += "floor"

        raw_model_cols += additive_regressors + multiplicative_regressors
        self.raw_model_df = df[raw_model_cols]
        self.set_y_scale()
        self.add_eligible_auto_seasonalities()

        # add seasonalities to df
        # TODO

        self.model_df = self._produce_model_matrix(df)
        self.data_assigned = True

    def add_eligible_auto_seasonalities(self):
        # auto determine seasonalities in Prophet Method
        first = self.raw_model_df["ds"].min()
        last = self.raw_model_df["ds"].max()
        range = last - first
        dt = self.raw_model_df["ds"].diff()
        min_dt = dt.iloc[dt.values.nonzero()[0]].min()

        # yearly if therea are >= 2 years of history
        if self.config.yearly_seasonality == "auto" and range >= pd.Timedelta(days=730):
            self.add_yearly_seasonality()

        # weekly if there are >= 2 weeks of history and there exists spacing < 7 days
        if self.config.weekly_seasonality == "auto" and (
            range >= pd.Timedelta(weeks=2) and min_dt < pd.Timedelta(weeks=1)
        ):
            self.add_weekly_seasonality()

        # daily if there are >= 2 days of history and there exists spacing < 1 day
        if self.config.daily_seasonality == "auto" and (
            range >= pd.Timedelta(days=2) and min_dt < pd.Timedelta(days=1)
        ):
            print(min_dt)
            self.add_daily_seasonality()

    def add_seasonality(
        self,
        name: str,
        peroid: float = 7,
        fourier_order: float = 3,
        mode: Literal["additive", "multiplicative"] = "additive",
    ):
        if name in [term.name for term in self.seasonalities]:
            raise ValueError(f"Seasonality term '{name}' already exists.")

        self.seasonalities.append(
            SeasonalityTermConfig(name=name, peroid=peroid, fourier_order=fourier_order, mode=mode)
        )

    def add_daily_seasonality(self, fourier_order: float = 4):
        self.add_seasonality("daily", peroid=1, fourier_order=fourier_order, mode=self.config.seasonality_mode)

    def add_weekly_seasonality(self, fourier_order: float = 3):
        self.add_seasonality("weekly", peroid=7, fourier_order=fourier_order, mode=self.config.seasonality_mode)

    def add_yearly_seasonality(self, fourier_order: float = 10):
        self.add_seasonality("yearly", peroid=365.25, fourier_order=fourier_order, mode=self.config.seasonality_mode)

    def validate_matrix(self, df: pd.DataFrame):
        """
        Validate that model cols are all there
        """
        if self.config.growth == "logistic":
            if "cap" not in df.columns:
                raise ValueError("Need to specify carring capacity in col 'cap'")
            if self.config.logistic_floor:
                if "floor" not in df.columns:
                    raise ValueError("floor enabled requires col 'floor'")

        missing_regressors = [r for r in self.additive_regressors + self.multiplicative_regressors if r not in df.cols]
        if missing_regressors:
            raise ValueError(f"Data does not contain regressors\n{missing_regressors}")

    def produce_model_matrix(self, df: pd.DataFrame):
        """
        This wrapped version of produce applies to future dataframes too. Need to validate.
        """
        self.validate_matrix(df)
        return self._produce_model_matrix(df)

    def _produce_model_matrix(self, df: pd.DataFrame):
        """
        Used in assign model matrix but made generic so we can apply to future df.
        """

        # ds must be a timestamp
        if not is_datetime64_any_dtype(df["ds"]):
            raise ValueError(f"Col 'ds' dtype {df['ds'].dtype} is not datetime")
        # scale y var
        df = self.transform_y(df)

        # seasonality uses days since epoch
        day_to_nanosec = 3600 * 24 * int(1e9)
        dates = df["ds"].to_numpy(dtype=np.int64) / day_to_nanosec
        df["t_seasonality"] = dates

        # add seasonality cols

        return df

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
