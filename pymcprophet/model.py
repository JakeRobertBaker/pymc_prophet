from typing import Literal
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import numpy as np
import datetime as dt
import re 

from pymcprophet.model_templates import BayesTSConfig, Feature, RegressorFeature, HolidayFeature, ModelSpecification, SeasonalityFeature
from pymcprophet.utils.make_holiday import make_holidays_df, get_country_holidays_class


class BayesTS:
    def __init__(
        self,
        config: BayesTSConfig,
        holidays=None,
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

    def add_holiday_country(self, country: str):
        # get raises Attribute error if country is not avaliable
        get_country_holidays_class(country)
        self.holiday_country = country

    def add_holiday(
        self,
        hol_df: pd.DataFrame,
        holiday_family: str = "user_specified_holiday",
        prior_scale: float | None = None,
        mode: Literal["additive", "multiplicative"] | None = None,
    ):
        """
        Add user specified holidays

        Args:
            hol_df pd.DataFrame with cols: ds, holiday, upper_window_lower_window
        """
        if self.data_assigned:
            raise ValueError("We do not support adding holidays after data is assigned. Do that before.")

        if not prior_scale:
            prior_scale = self.config.regressor_prior_scale

        if not mode:
            mode = self.config.regressor_mode

        regressor_prior_params = {"mu": 0, "sigma": prior_scale}

        if not is_datetime64_any_dtype(hol_df["ds"]):
            raise ValueError(f"Holiday 'ds' dtype {hol_df['ds'].dtype} is not datetime")

        # basic form is equivalent to window = 0
        for window_col in ["lower_window", "upper_window"]:
            if window_col not in hol_df.columns:
                hol_df[window_col] = 0

        for hol_name in hol_df["holiday"].unique():
            hdf = hol_df.query("holiday == @hol_name").copy()

            # define inclusive date range
            hdf["start_ds"] = hdf["ds"] - hdf["lower_window"] * dt.timedelta(days=1)
            hdf["end_ds"] = hdf["ds"] + hdf["upper_window"] * dt.timedelta(days=1)
            hdf["date_list"] = hdf.apply(lambda row: pd.date_range(row["start_ds"], row["end_ds"]).tolist(), axis=1)
            # map the list of time stamps to a list on unique dates
            hdates: list[dt.datetime] = list(set([timestamp.date() for row_date_list in hdf["date_list"] for timestamp in row_date_list]))
            
            holiday_short = hol_name.replace("_observed","")
            holiday_short = re.sub(pattern=r"[^a-zA-Z0-9]",repl="",string=holiday_short)
            holiday_short = re.sub(pattern=r" ",repl="_",string=holiday_short)

            self.model_spec.add_feature(
                holiday_short,
                HolidayFeature(
                    family_name=holiday_family, mode=mode, prior_kind="normal", prior_params=regressor_prior_params, dates=hdates
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
                family_name=regressor_family, mode=mode, prior_kind="normal", prior_params=regressor_prior_params, standardize=standardize
            ),
        )

    def assign_model_matrix(self, df: pd.DataFrame):
        """
        Assign the model data to model class instance.
        This is done once per model.
        Assigns things like scale params.
        """
        if self.data_assigned:
            raise ValueError("Data already assigned to model")

        # ensure basic cols and types are present
        self.validate_input_matrix(df)
        self.raw_model_df = df[self.get_input_model_cols()]

        # set scales, seasonalities, holidays
        self._set_y_scale()
        self._set_regressor_scales()
        self.determine_seasonalities()
        self.determine_country_holidays()

        # use class information to produce the model matrix
        self.model_df = self._produce_model_matrix(df)
        self.data_assigned = True

    def determine_country_holidays(self):
        if self.holiday_country:
            min_year = self.raw_model_df["ds"].dt.year.min()
            max_year = self.raw_model_df["ds"].dt.year.max() + self.config.forward_support_years
            hol_df = make_holidays_df(year_list=np.arange(min_year, max_year), country=self.holiday_country)
            self.add_holiday(hol_df, f"holiday_library_{self.holiday_country}")

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
        df = self._transform_y(df)
        df - self._transform_regressors(df)
        # TODO add transform all other vars may be done

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

            sin_df = pd.DataFrame(sin_terms.T, columns=sin_col_names, index=df.index)
            cos_df = pd.DataFrame(cos_terms.T, columns=cos_col_names, index=df.index)
            df = pd.concat([df, sin_df, cos_df], axis=1)

            for holiday_name, holiday_feature in self.model_spec.get_holiday_feature_dict().items():
                df[holiday_name] = df["ds"].dt.date.isin(holiday_feature.dates).astype(int)

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
                df[f"{reg_name}_std"] = (df[reg_name] - regressor.standardize_params["shift"]) / regressor.standardize_params["scale"]

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
