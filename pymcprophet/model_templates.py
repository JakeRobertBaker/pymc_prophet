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
    family_type: Literal["feature"] = "feature"

    def get_cols(self):
        return [feature_name for feature_name in self.features.keys()]

    def get_input_feature_cols(self):
        return [feature_name for feature_name, feature in self.features.items() if feature.feature_origin == "input"]

    def add_feature(self, feature_name: str, feature: Feature):
        if feature_name in self.features.keys():
            raise ValueError(f"A feature of of name {feature_name} has already been added to this family.")
        self.features = self.features | {feature_name: feature}


class _RegressorFamily(FeatureFamily):
    mode: Literal["additive", "multiplicative"]


class RegressorFamily(_RegressorFamily):
    family_type: Literal["regressor"] = "regressor"
    regressor_type: Literal["seasonal", "extra_regressor", "holiday"]


class SeasonalityFamily(_RegressorFamily):
    family_type: Literal["seasonal"] = "seasonal"
    period: float = Field(gt=0)
    fourier_order: int = Field(gt=0)


# each FeatureFamily lives in ModelSpecification ----------
class ModelSpecification(BaseModel, validate_assignment=True):
    feature_families: dict[str, FeatureFamily | RegressorFamily | SeasonalityFamily] = {}

    def add_family(self, family_name: str, feature_family: FeatureFamily):
        if family_name in self.feature_families.keys():
            raise ValueError(f"A feature family of name {family_name} has already been added.")
        self.feature_families = self.feature_families | {family_name: feature_family}

    def _get_feature_cols(
        self,
        family_types_filter: list[str] = ["seasonal", "regressor", "feature"],
        feature_origin_filter: list[str] = ["input", "generated"],
    ):
        cols = []
        for family_name, feature_family in self.feature_families.items():
            if feature_family.family_type in family_types_filter:
                for feature_name, feature in feature_family.features.items():
                    if feature.feature_origin in feature_origin_filter:
                        cols.append(feature_name)

        return cols
    
    def get_input_feature_cols(self):
        return self._get_feature_cols(feature_origin_filter=["input"])

    def get_feature_cols(self):
        return self._get_feature_cols(family_types_filter=["feature"])

    def get_regressor_cols(self):
        return self._get_feature_cols(family_types_filter=["regressor"])

    def get_seasonality_cols(self):
        return self._get_feature_cols(family_types_filter=["seasonal"])

    def get_seasonal_families(self):
        return {k: v for k, v in self.feature_families.items() if v.family_type == "seasonal"}
