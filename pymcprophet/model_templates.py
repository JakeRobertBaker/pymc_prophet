from pydantic import BaseModel, Field
from typing import Literal


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


# Prophet input args TODO
# changepoints=None, TODO
# n_changepoints=25, TODO
# changepoint_range=0.8,TODO
# holidays=None, -  let's do this as add regressor
# holidays_prior_scale=10.0, - doing as regressor prior scale
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
    mode: Literal["additive", "multiplicative"]
    bayes_params: dict[str, float] | None = None
    # TODO bayes params
    prior_kind: Literal["normal", "laplace"]
    prior_params: dict[str, float]


class RegressorFeature(_RegressorFeature):
    family_type: Literal["regressor"] = "regressor"
    regressor_type: Literal["extra_regressor", "holiday"] = "extra_regressor"
    standardize: Literal[True, False, "auto"]
    standardize_params: dict[Literal["scale", "shift"], float] | None = None

class SeasonalityFeature(_RegressorFeature):
    feature_origin: Literal["generated"] = "generated"
    family_type: Literal["seasonal"] = "seasonal"
    period: float = Field(gt=0)
    fourier_order: int = Field(gt=0)


#
class ModelSpecification(BaseModel, validate_assignment=True):
    features: dict[str, Feature | RegressorFeature | SeasonalityFeature] = {}

    def add_feature(self, feature_name: str, feature: Feature):
        if feature_name in self.features.keys():
            raise ValueError(f"A feature of of name {feature_name} has already been added to this family.")

        self.features = self.features | {feature_name: feature}

    def _get_features(
        self,
        family_types_filter: list[Literal["seasonal", "regressor", "feature"]] = None,
        feature_origin_filter: list[Literal["input", "generated"]] = None,
    ) -> dict[str, Feature | RegressorFeature | SeasonalityFeature]:
        eligible_feature_dict = self.features
        if family_types_filter:
            eligible_feature_dict = {
                feature_name: feature
                for feature_name, feature in eligible_feature_dict.items()
                if feature.family_type in family_types_filter
            }

        if feature_origin_filter:
            eligible_feature_dict = {
                feature_name: feature
                for feature_name, feature in eligible_feature_dict.items()
                if feature.feature_origin in feature_origin_filter
            }

        return eligible_feature_dict

    def get_cols(self):
        return list(self._get_features().keys())

    def get_input_feature_dict(self):
        return self._get_features(feature_origin_filter=["input"])

    def get_input_feature_cols(self):
        return list(self.get_input_feature_dict().keys())

    def get_seasonality_feature_dict(self) -> dict[str, SeasonalityFeature]:
        return self._get_features(family_types_filter=["seasonal"])

    def get_seasonality_feature_cols(self):
        return list(self.get_seasonality_feature_dict().keys())
    
    def get_regressor_feature_dict(self) -> dict[str, RegressorFeature]:
        return self._get_features(family_types_filter=["regressor"])

    def get_regressor_feature_cols(self):
        return list(self.get_regressor_feature_dict().keys())

