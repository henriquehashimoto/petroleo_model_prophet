#Importação das bibliotecas
import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import formaterData, formaterUniqueid, modelo
from sklearn.pipeline import Pipeline
import joblib
from joblib import load
from datetime import date, datetime
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go

#

#carregando os dados 
dados = pd.read_csv('./dados/df.csv')
dados_linha = dados.copy()

dados_linha['ds'] = pd.to_datetime(dados_linha['ds'], format="%Y-%m-%d")


############################# Streamlit ############################
st.markdown("<h1 style='text-align: center; '> Previsão de Petróleo - Modelo Prophet</h1>", unsafe_allow_html = True)

st.warning(f"Obs: A última data da base de dados é : {str(dados_linha['ds'].max())[:10]}")


#dateFinal = st.date_input("Prencha a data que quer que seja previsto")
#dateFinalTrat = pd.to_datetime(dateFinal, format="%Y-%m-%d")
#quantidade_dias = abs((dateFinalTrat - dados_linha['ds'].max()).days)

quantidade_dias = st.slider("Quantidade de dias de previsão (5 anos - em dias)", 30, 1825)



#print(dateFinalTrat)




def pipeline(df):
  pipeline = Pipeline([
      ('formater_Data', formaterData()),
      ('formater_Unique_id', formaterUniqueid())
  ])
  df_pipeline = pipeline.fit_transform(df)
  return df_pipeline

df_pipe = pipeline(dados)

#print(df_pipe.head())

if st.button('Enviar'):
    #df_result = modelo.prophet(quantidade_dias, df_pipe)
    #print("apertou o botao")
    #print(df_result.head())
  
        df_pipe = df_pipe.drop('unique_id', axis=1)

        #Treina
        modelo_prophet = Prophet(interval_width = 0.95, daily_seasonality = True)
        modelo_prophet.fit(df_pipe)

        #previsão futura
        future = modelo_prophet.make_future_dataframe(periods=quantidade_dias)

        # Unindo o forecast usando a previsão futura
        forecast = modelo_prophet.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        df_result = forecast.reset_index().merge(df_pipe, on=['ds'], how='left')

        fig1 = modelo_prophet.plot(forecast)

        st.write(fig1)

        #grafico
        grafico1 = plot_plotly(modelo_prophet,forecast)
        st.plotly_chart(grafico1)

        grafico2 = plot_components_plotly(modelo_prophet,forecast)
        st.plotly_chart(grafico2)

  