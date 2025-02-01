from typing import Literal
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import numpy as np
from pydantic import BaseModel


class Bayes_TS:
    def __init__(
        self,
        growth: Literal["linear", "logisitc", "flat"] = "linear",
        y_scale_type: Literal["absmax", "minmax"] = "absmax",
        logisitc_floor: bool = False,
        daily_seasonality: Literal["enabled", "disabled", "auto"] = "auto",
        weekly_seasonality: Literal["enabled", "disabled", "auto"] = "auto",
        yearly_seasonality: Literal["enabled", "disabled", "auto"] = "auto",
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: Literal["additive", "multiplicative"] = "additive",
        **kwargs,
    ):
        if growth != "logisitc" and logisitc_floor:
            raise ValueError("logisitc_floor=True requires growth='logistic'")

        self.growth = growth
        self.y_scale_type = y_scale_type
        self.logisitc_floor = logisitc_floor
        self.seasonality_prior_scale = float(seasonality_prior_scale)

        # add seasonalites
        self.seasonalities = {}

        if daily_seasonality == "enabled":
            self.add_daily_seasonality(mode=seasonality_mode)

        if weekly_seasonality == "enabled":
            self.add_weekly_seasonality(mode=seasonality_mode)

        if yearly_seasonality == "enabled":
            self.add_yearly_seasonality(mode=seasonality_mode)

        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality

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

        self.additive_regressors = additive_regressors
        self.multiplicative_regressors = multiplicative_regressors

        self.validate_matrix(df)

        raw_model_cols = ["ds", "y"]
        if self.growth == "logisitc":
            raw_model_cols += "cap"
            if self.logisitc_floor:
                raw_model_cols += "floor"

        raw_model_cols += additive_regressors + multiplicative_regressors
        self.raw_model_df = df[raw_model_cols]
        self.set_y_scale()

        # determine seasonalities
        # TODO auto

        # add seasonalities to df

        self.model_df = self._produce_model_matrix(df)
        self.data_assigned = True

    def add_seasonality(
        self,
        name: str,
        peroid: float = 7,
        fourier_order: float = 3,
        mode: Literal["additive", "multiplicative"] = "additive",
    ):
        self.seasonalities[name] = {"peroid": peroid, "fourier_order": fourier_order, "mode": mode}

    def add_daily_seasonality(
        self,
        peroid: float = 1,
        fourier_order: float = 4,
        mode: Literal["additive", "multiplicative"] = "additive",
    ):
        self.add_seasonality("daily", peroid, fourier_order, mode)

    def add_weekly_seasonality(
        self,
        peroid: float = 7,
        fourier_order: float = 3,
        mode: Literal["additive", "multiplicative"] = "additive",
    ):
        self.add_seasonality("daily", peroid, fourier_order, mode)

    def add_yearly_seasonality(
        self,
        peroid: float = 365.25,
        fourier_order: float = 10,
        mode: Literal["additive", "multiplicative"] = "additive",
    ):
        self.add_seasonality("daily", peroid, fourier_order, mode)

    def validate_matrix(
        self,
        df: pd.DataFrame,
    ):
        """
        Validate that model cols are all there
        """
        if self.growth == "logisitc" and "cap" not in df.columns:
            raise ValueError("Need to specify carring capacity in col 'cap'")

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

        if self.logisitc_floor:
            shift = df["floor"]
        else:
            shift = 0 if self.y_scale_type == "absmax" else self.raw_model_df["y"].min()

        df["y_trans"] = (df["y"] - shift) / self.scale

        return df

    def set_y_scale(self):
        """
        Set the scale term in transform:
            y(t) ->  ( y(t) - shift(t) ) / scale
        """

        if self.y_scale_type == "absmax":
            if self.logisitc_floor:
                self.scale = (self.raw_model_df["y"] - self.raw_model_df["floor"]).abs().max()
            else:
                self.scale = self.raw_model_df["y"].abs().max()

        if self.y_scale_type == "minmax":
            if self.logisitc_floor:
                self.scale = self.raw_model_df["cap"].max() - self.raw_model_df["floor"].min()
            else:
                self.scale = self.raw_model_df["y"].max() - self.raw_model_df["y"].min()

        self.y_scales_set = True
