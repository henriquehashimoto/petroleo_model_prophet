#Importação das bibliotecas
import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import formaterData, formaterUniqueid
from sklearn.pipeline import Pipeline
import joblib
from joblib import load
from datetime import date, datetime
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


st.set_page_config(page_title="Previsão preços petróleo", page_icon="📊")

#carregando os dados 
dados = pd.read_csv('./dados/df.csv')
dados = dados.loc[(dados['ds'] >= '2020-01-01')]

#print(dados.tail())

dados_linha = dados.copy()

dados_linha['ds'] = pd.to_datetime(dados_linha['ds'], format="%Y-%m-%d")


############################# Streamlit ############################
st.markdown("<h1 style='text-align: center; '> Previsão de Petróleo - Modelo Prophet</h1>", unsafe_allow_html = True)

st.warning(f"Obs: A última data da base de dados é : {str(dados_linha['ds'].max())[:10]}")


dateFinal = st.date_input("Prencha a data final da previsão")
dateFinalTrat = pd.to_datetime(dateFinal, format="%Y-%m-%d")
quantidade_dias = abs((dateFinalTrat - dados_linha['ds'].max()).days)

#quantidade_dias = st.slider("Quantidade de dias de previsão (1 ano - em dias)", 30, 365)



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
        df_pipe = df_pipe.drop('unique_id', axis=1)

        #Treina
        modelo_prophet = Prophet(interval_width = 0.95, daily_seasonality = True)
        modelo_prophet.fit(df_pipe)

        #previsão futura
        future = modelo_prophet.make_future_dataframe(periods=quantidade_dias)

        # Unindo o forecast usando a previsão futura
        forecast = modelo_prophet.predict(future)

        forecast = forecast.round(2)
        #print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

        df_result = forecast.reset_index().merge(df_pipe, on=['ds'], how='left')

        fig1 = modelo_prophet.plot(forecast,xlabel="Data",ylabel="Preço")

        st.write(fig1)

        #grafico
        grafico1 = plot_plotly(modelo_prophet,forecast,xlabel="Data",ylabel="Preço")
        st.plotly_chart(grafico1)

        grafico2 = plot_components_plotly(modelo_prophet,forecast)
        st.plotly_chart(grafico2)
        
        
        #-----------------------------
        # TESTE DO RESULTADO
        #-----------------------------
        df_result.fillna(0, inplace=True)
        mae = round(mean_absolute_error(df_result['y'], forecast['yhat'][-len(df_result):]),4)
        mse =  round(mean_squared_error(df_result['y'], forecast['yhat'][-len(df_result):]),4)
        rmse = round(np.sqrt(mse),4)
        
        # Precisao % , dentro de um limite aceitavel
        lim_erro = 0.05
        df_result["acerto"] = ((df_result["yhat"] / df_result["y"]) - 1) < lim_erro
        perc_accerto = round((df_result["acerto"].sum() / len(df_result)) * 100, 4)
        
        st.markdown(
        '''
        ---
        ## Avaliação do resultado do modelo apresentado
        
        - **MAE (Erro absoluto médio):** 
          - É a média absoluta dos erros entre as previsões e os valores reais.
          - Quanto menor o MAE, melhor é o desempenho do modelo.
          - Interpretação: Se o MAE for 5, por exemplo, isso significa que, em média, a previsão têm um erro absoluto médio de 5 unidades em relação aos valores reais.
        
        #####
        - **MSE (Erro Quadrático Médio):** É
          - É a média dos quadrados dos erros entre as previsões e os valores reais.
          - Quanto menor o MSE, melhor é o desempenho do modelo.
          - Interpretação: Se o MSE for 25, por exemplo, isso significa que, em média, o quadrado dos erros das suas previsões é 25 unidades.
          
        #####
        - **RMSE (Raiz do Erro Quadrático Médio):** 
          - É a raiz quadrada do MSE. Porém, mais facilmente interpretável do que o MSE.
          - Interpretação: Se o RMSE for 5, por exemplo, isso significa que, em média, seus erros de previsão têm uma raiz quadrada média de 5 unidades em relação aos valores reais.
        
        #####
        - **Percentual de acerto:** 
          - Quantos dias projetados o modelo ficou com até 95% de precisão.
        ###
        '''
        )
        
        st.write("### Resultado da avaliação:")
        st.write(f"Valor MAE (Erro Absoluto Médio): {mae}")
        st.write(f"Valor MSE (Erro Quadrático Médio): {mse}")
        st.write(f"Valor RMSE (Raiz do Erro Quadrático Médio): {rmse}")
        st.write(f"Percentual de acerto: {perc_accerto}%")

  