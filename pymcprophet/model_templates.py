from pydantic import BaseModel, Field, field_validator
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

    @field_validator("floor_logistic")
    def check_floor_logistic(cls, v, values):
        if v and values.get("growth") != "logistic":
            raise ValueError("floor_logistic=True requires growth='logistic'")
        return v


# TODO rarefactor into keyed lists


# an individual feature ----------
class Feature(BaseModel, validate_assignment=True):
    pass


class RegressorFeature(Feature):
    prior_params: dict[str, float] | None = None
    posterior_dist: list | None = None
    posterior_stats: dict | None = None


# each feature lives in a FeatureConfig ----------
class FeatureFamily(BaseModel, validate_assignment=True):
    features: dict[str, Feature] = {}

    def get_cols(self):
        return [col_name for col_name in self.features.keys()]


class RegressorFamily(FeatureFamily):
    regressor_type: Literal["seasonal", "extra_regressor", "holiday"]
    mode: Literal["additive", "multiplicative"]
    features: dict[str, RegressorFeature] = {}


class SeasonalityFamily(RegressorFamily):
    regressor_type: Literal["seasonal"] = "seasonal"
    period: float = Field(gt=0)
    fourier_order: int = Field(gt=0)


# each FeatureConfig lives in ModelSpecification ----------
class ModelSpecification(BaseModel, validate_assignment=True):
    feature_families: dict[str, FeatureFamily] = {}
    regressor_families: dict[str, RegressorFamily] = {}
    seasonality_families: dict[str, SeasonalityFamily] = {}

    def get_feature_cols(self):
        return [col for f_cols in [f_fam.get_cols() for f_fam in self.feature_families.values()] for col in f_cols]

    def get_regressor_cols(self):
        return [col for f_cols in [f_fam.get_cols() for f_fam in self.regressor_families.values()] for col in f_cols]

    def get_seasonality_cols(self):
        return [col for f_cols in [f_fam.get_cols() for f_fam in self.seasonality_families.values()] for col in f_cols]
