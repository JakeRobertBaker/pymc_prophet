from pydantic import BaseModel, Field
from typing import Literal
import datetime as dt


class BayesTSConfig(BaseModel):
    growth: Literal["linear", "logistic", "flat"] = "linear"
    y_scale_type: Literal["absmax", "minmax"] = "absmax"
    floor_logistic: bool = False
    daily_seasonality: Literal["enabled", "disabled", "auto"] = "auto"
    weekly_seasonality: Literal["enabled", "disabled", "auto"] = "auto"
    yearly_seasonality: Literal["enabled", "disabled", "auto"] = "auto"
    seasonality_prior_scale: float = Field(10.0, gt=0)
    seasonality_mode: Literal["additive", "multiplicative"] = "additive"
    regressor_prior_scale: float = Field(10.0, gt=0)
    regressor_mode: Literal["additive", "multiplicative"] = "additive"
    changepoints: list[dt.date] = None
    n_changepoints: int = Field(25, gt=0)
    changepoint_range: float = Field(0.8, gt=0)
    changepoint_prior_scale: float = Field(0.05, gt=0)
    forward_support_years: int = Field(45, gt=0)


# Prophet input args TODO
# changepoints=None, TODO
# n_changepoints=25, TODO
# changepoint_range=0.8,TODO
# changepoint_prior_scale=0.05, TODO
# mcmc_samples=0,
# interval_width=0.80,
# uncertainty_samples=1000,
# stan_backend=None,
# scaling: str = 'absmax', TODO check if this mixes with regressors,think not and just need standardize in add regressor method.
# holidays_mode=None,


# an individual feature ----------
class Feature(BaseModel, validate_assignment=True):
    feature_origin: Literal["input", "generated"] = "input"
    family_type: Literal["feature"] = "feature"
    family_name: str


class _RegressorFeature(Feature):
    family_type: str
    mode: Literal["additive", "multiplicative"]
    prior_kind: Literal["normal", "laplace"]
    prior_params: dict[str, float]


class RegressorFeature(_RegressorFeature):
    family_type: Literal["regressor"] = "regressor"
    standardize: Literal[True, False, "auto"]
    standardize_params: dict[Literal["scale", "shift"], float] | None = None


class HolidayFeature(_RegressorFeature):
    feature_origin: Literal["generated"] = "generated"
    family_type: Literal["holiday"] = "holiday"
    dates: list[dt.date]


class SeasonalityFeature(_RegressorFeature):
    feature_origin: Literal["generated"] = "generated"
    family_type: Literal["seasonal"] = "seasonal"
    period: float = Field(gt=0)
    fourier_order: int = Field(gt=0)


class ModelSpecification(BaseModel, validate_assignment=True):
    features: dict[str, Feature | RegressorFeature | HolidayFeature | SeasonalityFeature] = {}

    def add_feature(self, feature_name: str, feature: Feature):
        if feature_name in self.features.keys():
            raise ValueError(f"A feature of of name {feature_name} has already been added to this family.")

        self.features = self.features | {feature_name: feature}

    def _get_feature_dict(self, filter_dict: dict[str, list[str] | str] = {}, required_properties: list[str] = []) -> dict[str, Feature]:
        eligible_feature_dict = self.features

        for property_name, property_filter in filter_dict.items():
            property_filter = [property_filter] if isinstance(property_filter, str) else property_filter
            eligible_feature_dict = {
                feature_name: feature
                for feature_name, feature in eligible_feature_dict.items()
                if getattr(feature, property_name) in property_filter
            }
        for property in required_properties:
            eligible_feature_dict = {
                feature_name: feature for feature_name, feature in eligible_feature_dict.items() if hasattr(feature, property)
            }
        return eligible_feature_dict

    def get_cols(self) -> list[str]:
        return list(self._get_feature_dict({}).keys())

    def get_input_feature_dict(self) -> dict[str, Feature]:
        return self._get_feature_dict({"feature_origin": "input"})

    def get_seasonality_feature_dict(self) -> dict[str, SeasonalityFeature]:
        return self._get_feature_dict({"family_type": "seasonal"})

    def get_regressor_feature_dict(self) -> dict[str, RegressorFeature]:
        return self._get_feature_dict({"family_type": "regressor"})

    def get_holiday_feature_dict(self) -> dict[str, HolidayFeature]:
        return self._get_feature_dict({"family_type": "holiday"})

    def get_additive_feature_dict(self) -> dict[str, Feature]:
        return self._get_feature_dict({"mode": "additive"})

    def get_multiplicative_feature_dict(self) -> dict[str, Feature]:
        return self._get_feature_dict({"mode": "multiplicative"})

    def get_bayes_feature_dict(self, mode: Literal["additive", "multiplicative", None] = None) -> dict[str, _RegressorFeature]:
        filter_dict = {"mode": mode} if mode else {}
        return self._get_feature_dict(filter_dict=filter_dict, required_properties=["prior_params"])

    def get_input_feature_cols(self) -> list[str]:
        return list(self.get_input_feature_dict().keys())

    def get_seasonality_feature_cols(self) -> list[str]:
        return list(self.get_seasonality_feature_dict().keys())

    def get_regressor_feature_cols(self) -> list[str]:
        return list(self.get_regressor_feature_dict().keys())

    def get_holiday_feature_cols(self) -> list[str]:
        return list(self.get_holiday_feature_dict().keys())

    def get_additive_feature_cols(self) -> list[str]:
        return list(self.get_additive_feature_dict().keys())

    def get_multiplicative_feature_cols(self) -> list[str]:
        return list(self.get_multiplicative_feature_dict().keys())

    def get_bayes_feature_cols(self, mode: Literal["additive", "multiplicative", None] = None) -> list[str]:
        return list(self.get_bayes_feature_dict(mode).keys())
