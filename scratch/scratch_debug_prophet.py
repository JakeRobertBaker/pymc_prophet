from prophet import Prophet
import pandas as pd


df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
df['ds'] = pd.to_datetime(df['ds'])

playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))

train_df = df.query("ds.dt.year <= 2014")
test_df = df.query("ds.dt.year > 2014")

prophet_model = Prophet(holidays=holidays)
prophet_model.add_country_holidays(country_name='US')
# prophet_model.add_regressor('nfl_sunday')
prophet_model.fit(train_df)

# country = 'US'
# year_list = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
# province = None
# state = None
