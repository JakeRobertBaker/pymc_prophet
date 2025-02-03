from idna import valid_contextj
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


# an individual feature ----------
class Feature(BaseModel, validate_assignment=True):
    feature_origin: Literal["input", "generated"] = "input"


class RegressorFeature(Feature):
    prior_params: dict[str, float] | None = None
    posterior_dist: list | None = None
    posterior_stats: dict | None = None


# each feature lives in a FeatureFamily ----------
class FeatureFamily(BaseModel, validate_assignment=True):
    features: dict[str, Feature] = {}

    def get_cols(self):
        return [col_name for col_name in self.features.keys()]

    def add_feature(self, feature_name: str, feature: Feature):
        self.features = self.features | {feature_name: feature}


class RegressorFamily(FeatureFamily):
    regressor_type: Literal["seasonal", "extra_regressor", "holiday"]
    mode: Literal["additive", "multiplicative"]
    features: dict[str, RegressorFeature] = {}


class SeasonalityFamily(RegressorFamily):
    regressor_type: Literal["seasonal"] = "seasonal"
    period: float = Field(gt=0)
    fourier_order: int = Field(gt=0)


# each FeatureFamily lives in ModelSpecification ----------
class ModelSpecification(BaseModel, validate_assignment=True):
    feature_families: dict[str, FeatureFamily] = {}
    regressor_families: dict[str, RegressorFamily] = {}
    seasonality_families: dict[str, SeasonalityFamily] = {}

    def add_family(
        self, family_type: Literal["feature", "regressor", "seasonal"], family_name: str, feature_family: FeatureFamily
    ):
        if family_type == "feature":
            self.feature_families = self.feature_families | {family_name: feature_family}
        elif family_type == "regressor":
            self.regressor_families = self.regressor_families | {family_name: feature_family}
        elif family_type == "seasonal":
            self.seasonality_families = self.seasonality_families | {family_name: feature_family}
        else:
            raise ValueError('family type must be in "feature", "regressor", "seasonal"')

    def add_feature_family(self, family_name: str, feature_family: FeatureFamily):
        self.add_family(family_type="feature", family_name=family_name, feature_family=feature_family)

    def add_regressor_family(self, family_name: str, feature_family: FeatureFamily):
        self.add_family(family_type="regressor", family_name=family_name, feature_family=feature_family)

    def add_seasonal_family(self, family_name: str, feature_family: FeatureFamily):
        self.add_family(family_type="seasonal", family_name=family_name, feature_family=feature_family)

    def get_feature_cols(self):
        return [col for f_cols in [f_fam.get_cols() for f_fam in self.feature_families.values()] for col in f_cols]

    def get_regressor_cols(self):
        return [col for f_cols in [f_fam.get_cols() for f_fam in self.regressor_families.values()] for col in f_cols]

    def get_seasonality_cols(self):
        return [col for f_cols in [f_fam.get_cols() for f_fam in self.seasonality_families.values()] for col in f_cols]
