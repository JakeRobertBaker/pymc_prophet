import datetime as dt
import re
from typing import Literal

import numpy as np
import pandas as pd
from pandas import Timedelta, Timestamp
from pandas.api.types import is_datetime64_any_dtype

from pymc_prophet.model_config import (
    BayesTSConfig,
    Feature,
    HolidayFeature,
    ModelSpecification,
    RegressorFeature,
    SeasonalityFeature,
)
from pymc_prophet.utils.make_holiday import get_country_holidays_class, make_holidays_df


class BayesTS:
    def __init__(
        self,
        config: BayesTSConfig,
    ):
        self.config = config
        self.model_spec = ModelSpecification()

        if config.growth == "logistic":
            self.model_spec.add_feature("cap", Feature(family_name="logistic"))
            if config.floor_logistic:
                self.model_spec.add_feature("floor", Feature(family_name="logistic"))

        self.data_assigned = False
        self.y_scales_set = False
        self.seasonality_families = {}
        self.holiday_country = None

    def add_holiday_country(
        self,
        country: str,
    ):
        # get raises Attribute error if country is not avaliable
        get_country_holidays_class(country)
        # property sets off logic in determine holiday function
        self.holiday_country = country

    def add_holiday(
        self,
        holidays_df: pd.DataFrame,
        holiday_family: str = "user_specified_holiday",
        prior_scale: float | None = None,
        mode: Literal["additive", "multiplicative"] | None = None,
        separate_lags: bool = True,
    ):
        """
        Add user specified holidays

        Args:
            holidays_df pd.DataFrame with cols: ds, holiday, upper_window_lower_window
        """
        if self.data_assigned:
            raise ValueError("We do not support adding holidays after data is assigned. Do that before.")

        if not prior_scale:
            prior_scale = self.config.regressor_prior_scale

        if not mode:
            mode = self.config.regressor_mode

        regressor_prior_params = {"mu": 0, "sigma": prior_scale}

        if not is_datetime64_any_dtype(holidays_df["ds"]):
            raise ValueError(f"Holiday 'ds' dtype {holidays_df['ds'].dtype} is not datetime")

        # basic form is equivalent to window = 0
        for window_col in ["lower_window", "upper_window"]:
            if window_col not in holidays_df.columns:
                holidays_df[window_col] = 0

        for hol_name, hol_df in holidays_df.groupby("holiday"):
            dates = []
            holiday_short = re.sub(pattern=r"\s+", repl="_", string=hol_name)
            holiday_short = re.sub(pattern=r"[^\w]", repl="", string=holiday_short).lower()

            # for every permitted lag get the dates
            for lag in range(-hol_df["lower_window"].max(), hol_df["upper_window"].max() + 1):
                lag_dates = hol_df.query(" lower_window >= -@lag and upper_window >= @lag ")["ds"].unique() - lag * dt.timedelta(
                    days=1
                )
                # treat the lags as separate cols
                if separate_lags:
                    holiday_short_delim = holiday_short if lag == 0 else f"{holiday_short}_lag_{lag}"
                    self.model_spec.add_feature(
                        holiday_short_delim,
                        HolidayFeature(
                            family_name=holiday_family,
                            mode=mode,
                            prior_kind="normal",
                            prior_params=regressor_prior_params,
                            dates=lag_dates,
                        ),
                    )
                # else treat all lags as a single col
                else:
                    dates += lag_dates

            # after for loop just add one col for that holiday
            if not separate_lags:
                dates = list(set(dates))
                self.model_spec.add_feature(
                    holiday_short,
                    HolidayFeature(
                        family_name=holiday_family,
                        mode=mode,
                        prior_kind="normal",
                        prior_params=regressor_prior_params,
                        dates=dates,
                    ),
                )

    def add_regressor(
        self,
        reg_name: str,
        prior_scale: float | None = None,
        mode: Literal["additive", "multiplicative"] | None = None,
        regressor_family: str | None = None,
        standardize: Literal[True, False, "auto"] = "auto",
    ):
        if self.data_assigned:
            raise ValueError("We do not support adding regressors after data is assigned. Do that before.")

        if not prior_scale:
            prior_scale = self.config.regressor_prior_scale

        if not mode:
            mode = self.config.regressor_mode

        if not regressor_family:
            regressor_family = mode + "_regressors"

        regressor_prior_params = {"mu": 0, "sigma": prior_scale}
        self.model_spec.add_feature(
            reg_name,
            RegressorFeature(
                family_name=regressor_family,
                mode=mode,
                prior_kind="normal",
                prior_params=regressor_prior_params,
                standardize=standardize,
            ),
        )

    def assign_model_matrix(self, raw_df: pd.DataFrame):
        """
        Assign the model data to model class instance.
        This is done once per model.
        Assigns things like scale params.
        """
        if self.data_assigned:
            raise ValueError("Data already assigned to model")

        # ensure basic cols and types are present
        self.validate_input_matrix(raw_df)
        self.raw_model_df = raw_df[self.get_input_model_cols()].copy()
        self.train_ds_start: Timestamp = self.raw_model_df["ds"].min()
        self.train_ds_end: Timestamp = self.raw_model_df["ds"].max()
        self.ds_scale: Timedelta = self.train_ds_end - self.train_ds_start

        # set scales, seasonalities, holidays
        self._set_y_scale()
        self._set_regressor_scales()
        self.determine_seasonalities()
        self.determine_country_holidays()

        # use class information to produce the model matrix
        self.model_df = self._produce_model_matrix(raw_df)

        # once times are defined we can set the t values of the changepoints
        if not self.config.changepoints:
            self._auto_set_changepoints()
        self.t_values = [(cp_date - self.train_ds_start.date()) / self.ds_scale for cp_date in self.config.changepoints]

        self.data_assigned = True

        # some objects that are used during modeling
        self.additive_feature_dict = self.model_spec.get_bayes_feature_dict(mode="additive")
        self.multiplicative_feature_dict = self.model_spec.get_bayes_feature_dict(mode="multiplicative")

    def _produce_modelling_data(self, model_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Product the arrays fed into pm.Data during modelling

        Args:
            model_df (pd.DataFrame): processed model matrix
        """
        t_np = model_df["t"].to_numpy()
        X_additive = model_df[self.additive_feature_dict.keys()].to_numpy()
        X_multiplicative = model_df[self.multiplicative_feature_dict.keys()].to_numpy()

        return t_np, X_additive, X_multiplicative

    def fit(self, raw_df: pd.DataFrame, **kwargs):
        """
        Fit the model to the raw dataframe
        """
        self.assign_model_matrix(raw_df)
        self._fit(raw_df, **kwargs)

    def _fit(self, raw_df: pd.DataFrame, **kwargs):
        """
        Abstract method to fit the model. Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement the _fit method.")

    def _auto_set_changepoints(self):
        """
        select n_changpoints of the from the first changepoint_range of ds values
        """
        cp_idx_max = int(self.config.changepoint_range * len(self.model_df))
        cp_idx = np.linspace(0, cp_idx_max, self.config.n_changepoints + 1)[1:]
        ds_series: pd.Series[pd.Timestamp] = self.model_df["ds"].iloc[cp_idx]
        self.config.changepoints = ds_series.dt.date.values.tolist()

    def set_changepoints(self, changepoints: list[str | dt.date]):
        """
        Set the model changepoints as a list of dates
        """
        if self.config.changepoints:
            raise ValueError("Changepoints already set. Set them before fit.")

        if isinstance(changepoints[0], str):
            changepoints = pd.to_datetime(changepoints).to_pydatetime().tolist()

        self.config.changepoints = changepoints

    def get_t_change(self):
        if not self.data_assigned():
            raise ValueError("Need to assign model matrix before accessing cp t values.")

    def determine_country_holidays(self):
        if self.holiday_country:
            min_year = self.raw_model_df["ds"].dt.year.min()
            # forward support is required for out of sample predictor values
            max_year = self.raw_model_df["ds"].dt.year.max() + self.config.forward_support_years
            holidays_df = make_holidays_df(year_list=np.arange(min_year, max_year), country=self.holiday_country)
            # if the hoiday never exists in train time discard it
            holidays_to_exclude = (  # noqa: F841
                holidays_df.groupby("holiday")
                .agg({"ds": "min"})
                .query("ds > @self.train_ds_end")
                .reset_index()["holiday"]
                .unique()
                .tolist()
            )

            holidays_df = holidays_df.query("~ holiday.isin(@holidays_to_exclude)").reset_index()
            self.add_holiday(holidays_df, f"holiday_library_{self.holiday_country}", separate_lags=False)

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
        if not self.data_assigned:
            raise ValueError("To obtain an out of sample model matrix the in sample matrix must be assigned first.")

        self.validate_input_matrix(df)
        return self._produce_model_matrix(df)

    def _produce_model_matrix(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Used in assign model matrix but made generic so we can apply to future df.
        """
        df = input_df.copy()
        # scale variables
        df = self._transform_y(df)
        df = self._transform_regressors(df)
        df = self._transform_t(df)

        # add seasonality cols
        df = self._produce_seasonality_matrix(df)

        # add the holidays
        for holiday_name, holiday_feature in self.model_spec.get_holiday_feature_dict().items():
            df[holiday_name] = df["ds"].dt.date.isin(holiday_feature.dates).astype(int)

        return df

    def _produce_seasonality_matrix(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
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

            # When we do this for the first time add these created cols to the model spec
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

            sin_df = pd.DataFrame(sin_terms.T, columns=sin_col_names, index=df.index)
            cos_df = pd.DataFrame(cos_terms.T, columns=cos_col_names, index=df.index)
            df = pd.concat([df, sin_df, cos_df], axis=1)

        return df

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

    def _transform_t(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
        df["t"] = (df["ds"] - self.train_ds_start) / self.ds_scale
        return df

    def _transform_y(self, input_df: pd.DataFrame) -> pd.DataFrame:
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

        df["y_trans"] = (df["y"] - shift) / self.y_scale

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
                self.y_scale = (self.raw_model_df["y"] - self.raw_model_df["floor"]).abs().max()
            else:
                self.y_scale = self.raw_model_df["y"].abs().max()

        if self.config.y_scale_type == "minmax":
            if self.config.floor_logistic:
                self.y_scale = self.raw_model_df["cap"].max() - self.raw_model_df["floor"].min()
            else:
                self.y_scale = self.raw_model_df["y"].max() - self.raw_model_df["y"].min()

        self.y_scales_set = True

    def _transform_regressors(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()

        for reg_name, regressor in self.model_spec.get_regressor_feature_dict().items():
            if regressor.standardize:
                df[f"{reg_name}_std"] = (df[reg_name] - regressor.standardize_params["shift"]) / regressor.standardize_params[
                    "scale"
                ]

        return df

    def _set_regressor_scales(self):
        for reg_name, regressor in self.model_spec.get_regressor_feature_dict().items():
            uinique_reg_vals = set(self.raw_model_df[reg_name].unique())

            # Never standardize single valued regressor
            if len(uinique_reg_vals) <= 1:
                regressor.standardize = False

            # 'auto' setting only denies binary variables
            if regressor.standardize == "auto":
                if uinique_reg_vals == {1, 0}:
                    regressor.standardize = False
                else:
                    regressor.standardize = True

            if regressor.standardize:
                shift = self.raw_model_df[reg_name].mean()
                scale = self.raw_model_df[reg_name].std()
                regressor.standardize_params = {"scale": scale, "shift": shift}

    def get_input_model_cols(self) -> list[str]:
        """
        Generate a list of column names required in input matrix.
        """
        return ["ds", "y"] + self.model_spec.get_input_feature_cols()
