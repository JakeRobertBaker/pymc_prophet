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


# an individual feature ----------
class Feature(BaseModel, validate_assignment=True):
    col_name: str


class RegressorFeature(Feature):
    prior_params: dict[str, float] | None = None
    posterior_dist: list | None = None
    posterior_stats: dict | None = None


# each feature lives in a FeatureConfig ----------
class FeatureConfig(BaseModel, validate_assignment=True):
    name: str
    features: list[Feature] = []

    def get_cols(self):
        return [feature.col_name for feature in self.features]


class RegressorFeatureConfig(FeatureConfig):
    name: str
    regressor_type: Literal["seasonal", "extra_regressor", "holiday"]
    mode: Literal["additive", "multiplicative"]
    features: list[RegressorFeature] = []


class SeasonalityConfig(RegressorFeatureConfig):
    regressor_type: Literal["seasonal"] = "seasonal"
    period: float = Field(gt=0)
    fourier_order: int = Field(gt=0)


# each FeatureConfig lives in ModelSpecification ----------
class ModelSpecification(BaseModel, validate_assignment=True):
    feature_configs: list[FeatureConfig]
    regressor_configs: list[RegressorFeatureConfig]
    seasonality_configs: list[SeasonalityConfig]

    def get_feature_cols(self):
        return [col for f_cols in [f_cfg.get_cols() for f_cfg in self.feature_configs] for col in f_cols]

    def get_regressor_cols(self):
        return [col for f_cols in [f_cfg.get_cols() for f_cfg in self.regressor_configs] for col in f_cols]

    def get_seasonality_cols(self):
        return [col for f_cols in [f_cfg.get_cols() for f_cfg in self.seasonality_configs] for col in f_cols]
