
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from prophet import Prophet

# Classes para pipeline

class formaterData(BaseEstimator, TransformerMixin):
  def __init__(self, data=['ds']):
    self.data = data

  def fit(self, df):
    return self

  def transform(self, df):
    if (set(self.data).issubset(df.columns)):
      #df['ds'] = pd.to_datetime(df['Data'])
      df['ds'] = pd.to_datetime(df['ds'], format="%Y-%m-%d")
      #df.index = pd.to_datetime(df.Data, format = "%d.%m.%Y")
      #df.drop("Data", inplace=True, axis=1)
      return df
    else:
      print('Informação de Data não esta no DataFrame')
      return df

class formaterUniqueid(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, df):
    return self

  def transform(self, df):
      df['unique_id'] = 'PETROLEO'
      return df
  
class modelo():
    def prophet(periods, df_pipe):
        df_pipe = df_pipe.drop('unique_id', axis=1)

        #Treina
        modelo_prophet = Prophet(interval_width = 0.95, daily_seasonality = True)
        modelo_prophet.fit(df_pipe)

        #previsão futura
        future = modelo_prophet.make_future_dataframe(periods=periods)

        # Unindo o forecast usando a previsão futura
        forecast = modelo_prophet.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        df_result = forecast.reset_index().merge(df_pipe, on=['ds'], how='left')

        fig1 = modelo_prophet.plot(forecast)

        return df_result
  