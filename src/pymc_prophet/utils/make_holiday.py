# Copy of the make holiday function taken from the prophet repo
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import holidays
import pandas as pd


def get_country_holidays_class(country):
    """Get class for a supported country.

    Parameters
    ----------
    country: country code

    Returns
    -------
    A valid country holidays class
    """
    substitutions = {
        "TU": "TR",  # For compatibility with Turkey as 'TU' cases.
    }

    country = substitutions.get(country, country)
    if not hasattr(holidays, country):
        raise AttributeError(f"Holidays in {country} are not currently supported!")

    return getattr(holidays, country)


def make_holidays_df(year_list, country, province=None, state=None) -> pd.DataFrame:
    """Make dataframe of holidays for given years and countries

    Parameters
    ----------
    year_list: a list of years
    country: country name

    Returns
    -------
    Dataframe with 'ds' and 'holiday', which can directly feed
    to 'holidays' params in Prophet
    """
    country_holidays = get_country_holidays_class(country)
    holidays = country_holidays(expand=False, language="en_US", subdiv=province, years=year_list)

    holidays_df = pd.DataFrame(
        [(date, holidays.get_list(date)) for date in holidays],
        columns=["ds", "holiday"],
    )
    holidays_df = holidays_df.explode("holiday")
    holidays_df.reset_index(inplace=True, drop=True)
    holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])

    return holidays_df
