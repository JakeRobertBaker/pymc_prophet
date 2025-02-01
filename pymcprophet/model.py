from typing import Literal
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import numpy as np


class Bayes_TS:
    def __init__(
        self,
        growth: Literal["linear", "logisitc", "flat"] = "flat",
        y_scale_type: Literal["absmax", "minmax"] = "absmax",
        n_weekly_terms: int = 3,
        n_yearly_terms: int = 10,
        logisitc_floor: bool = False,
        **kwargs,
    ):
        self.growth = growth
        self.y_scale_type = y_scale_type
        self.n_weekly_terms = n_weekly_terms
        self.n_yearly_terms = n_yearly_terms

        if growth != "logisitc" and logisitc_floor:
            raise ValueError("logisitc_floor=True requires growth='logistic'")

        self.logisitc_floor = logisitc_floor

        self.data_assigned = False

    def assign_model_matrix(
        self,
        df: pd.DataFrame,
        additive_regressors: list[str] | None,
        multiplicative_regressors: list[str] | None,
    ):
        """
        Assign the model data to model class instance.
        This is done once per model.
        Assigns things like scale params.
        """
        if self.data_assigned:
            ValueError("Data already assigned to model")

        if self.growth == "logisitc" and "cap" not in df.columns:
            raise ValueError("logistic requires 'cap' col specifying carry capacity")

        missing_regressors = [
            regressor
            for regressor in additive_regressors + multiplicative_regressors
            if regressor not in df.cols
        ]

        if missing_regressors:
            raise ValueError(f"data does not contain regressors\n{missing_regressors}")

        raw_model_cols = ["ds", "y"]
        if self.growth == "logisitc":
            raw_model_cols += "cap"
            if self.logisitc_floor:
                raw_model_cols += "floor"

        raw_model_cols += additive_regressors + multiplicative_regressors

        self.raw_model_df = df[raw_model_cols]

        self.data_assigned = True

    def produce_model_matrix(self, df: pd.DataFrame):
        """
        Used in assign model matrix but made generic so we can apply to future df.
        """
        # ds must be a timestamp
        if not is_datetime64_any_dtype(df["ds"]):
            raise ValueError(f"Col 'ds' dtype {df['ds'].dtype} is not datetime")

        # seasonality uses days since epoch
        day_to_nanosec = 3600 * 24 * int(1e9)
        dates = df["ds"].to_numpy(dtype=np.int64) / day_to_nanosec
        df["t_seasonality"] = dates

        return df

    def scale_y(self, df: pd.DataFrame):
        pass

    def set_y_scale(self):
        """
        set the affine transform that does:
            y(t) -> scale * y(t) - shift(t)
        """

        if self.y_scale_type == "absmax":
            # scale is the shifted data by abs max
            self.shift = self.raw_model_df["floor"] if self.logisitc_floor else 0
            self.scale = (self.raw_model_df["y"] - self.shift()).abs().max()

        if self.y_scale_type == "minmax":
            if self.logisitc_floor:
                self.shift = self.raw_model_df["floor"]
                self.scale = (
                    self.raw_model_df["cap"].max() - self.raw_model_df["floor"].min()
                )
