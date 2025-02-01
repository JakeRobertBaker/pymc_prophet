import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import numpy as np


class Bayes_TS:
    def __init__(self, **kwargs):
        self.n_weekly_terms = kwargs.get("n_weekly_terms", 3)
        self.n_yearly_terms = kwargs.get("n_yearly_terms", 10)

    def process_df(self, df: pd.DataFrame):
        # ds must be a timestamp
        if not is_datetime64_any_dtype(df["ds"]):
            raise ValueError(f"Col 'ds' dtype {df['ds'].dtype} is not datetime")

        # seasonality uses days since epoch
        day_to_nanosec = 3600 * 24 * int(1e9)
        dates = df["ds"].to_numpy(dtype=np.int64) / day_to_nanosec
        df["t_seasonality"] = dates

        return df
