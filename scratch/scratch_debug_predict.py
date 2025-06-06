# from prophet import Prophet
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from pymc_prophet.model import BayesTS, BayesTSConfig

print(f"Running on PyMC v{pm.__version__}")

# Initialize random number generator
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

df = pd.read_csv("https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv")
df["ds"] = pd.to_datetime(df["ds"])

np.random.default_rng(RANDOM_SEED)

playoffs = pd.DataFrame(
    {
        "holiday": "playoff",
        "ds": pd.to_datetime(
            [
                "2008-01-13",
                "2009-01-03",
                "2010-01-16",
                "2010-01-24",
                "2010-02-07",
                "2011-01-08",
                "2013-01-12",
                "2014-01-12",
                "2014-01-19",
                "2014-02-02",
                "2015-01-11",
                "2016-01-17",
                "2016-01-24",
                "2016-02-07",
            ]
        ),
        "lower_window": 0,
        "upper_window": 1,
    }
)
superbowls = pd.DataFrame(
    {
        "holiday": "superbowl",
        "ds": pd.to_datetime(["2010-02-07", "2014-02-02", "2016-02-07"]),
        "lower_window": 0,
        "upper_window": 1,
    }
)
holidays = pd.concat((playoffs, superbowls))


def nfl_sunday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6 and (date.month > 8 or date.month < 2):
        return 1
    else:
        return 0


df["nfl_sunday"] = df["ds"].apply(nfl_sunday)

train_df = df.query("ds.dt.year <= 2014")
test_df = df.query("ds.dt.year > 2014")

ts_config = BayesTSConfig()
ts_model = BayesTS(ts_config)
ts_model.add_holiday_country("US")
ts_model.add_holiday(holidays)
ts_model.add_regressor("nfl_sunday", mode="additive")

# ts_model.assign_model_matrix(train_df)

ts_model.fit(train_df, random_seed=rng)

ts_model.predict(test_df)
