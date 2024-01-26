
##################################################
# TECH CHALLENGE 04 

# - Base de dados - dados históricos do preco de petroleo brent
# - Tarefas: 
#   - Storytelling que traga insights relevantes sobre a variação do preço do petróleo, como situações geopolíticas, crises econômicas, etc
#   - Pelo menos 4 insights
# Criar um modelo de Machine Learning que faça a previsão do preço do petróleo diariamente
# Faça um MVP do seu modelo em produção utilizando o Streamlit.

# Ideia: 
#   Ter 2 paginas, 1 para dados históricos e outra para o modelo

##################################################

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import altair as alt
from statsmodels.tsa.seasonal import seasonal_decompose


##----------------------------------------------------------------------------------
# Carregando dados e introducao
##----------------------------------------------------------------------------------
# Importando dados 
dados = pd.read_csv("./dados/historico_petroleo.csv")
dados["data"] = pd.to_datetime(dados["data"])
dados["preco"] = dados["preco_petroleo_bruto_brent"]

# Config iniciais
st.set_page_config(page_title="Dados histórico", page_icon="📈")
st.write("# Veja abaixo os dados históricos e uma Analise Exploratória")
st.markdown(
'''
### Pro tip:
- É possível filtrar parte do gráfico, basta clicar dentro dele e arrastar selecionando o período especifico
'''
)


##----------------------------------------------------------------------------------
# Criar gráfico de linha interativo
##----------------------------------------------------------------------------------
plt.figure(figsize=(20,6))
fig = px.line(dados, x="data", y="preco", title="Variação do Preço do Petróleo")
st.plotly_chart(fig)


#----------------------------------------------------------------------------------
# Grafico filtrado 
#----------------------------------------------------------------------------------
#st.markdown(
#'''
#### 
#### 
#### Ou então: selecione a data exata que deseja filtrar
#'''
#)
#start_date = st.date_input("Selecione a data inicio", min(dados["data"]))
#end_date = st.date_input("Selecione a data fim", max(dados["data"]))
#start_date = pd.to_datetime(start_date)
#end_date = pd.to_datetime(end_date)
#dados_filt = dados[(dados["data"] >= start_date) & (dados["data"] <= end_date)]
#
#
#chart = alt.Chart(dados_filt).mark_line().encode(
#    x='data:T',
#    y='preco:Q',
#).properties(
#    width=1000,
#    height=600
#)
## Exibindo o gráfico no Streamlit
#st.altair_chart(chart, use_container_width=True)


#----------------------------------------------------------------------------------
# ANALISE EXPLORATÓRIA DE DADOS
#----------------------------------------------------------------------------------
st.markdown(
'''
---
## Analise Exploratória de Dados (EDA)
'''
)

st.write("### **Primeiro, comece setando o periodo a ser analisado (entre 1987-2014)**")
start_date2 = st.date_input("Selecione a data de inicio", min(dados["data"]))
end_date2 = st.date_input("Selecione a data de fim", max(dados["data"]))
start_date2 = pd.to_datetime(start_date2)
end_date2 = pd.to_datetime(end_date2)
dados_filt2 = dados[(dados["data"] >= start_date2) & (dados["data"] <= end_date2)]


#----------------------
# Estatistica basica
#----------------------
st.write("### **Estatística básica**")
st.table(dados_filt2["preco"].describe())


#----------------------
# Distribuicao dos precos ao longo do tempo
#----------------------
st.write("### **Distribuição de preço por quantidade de dias**")

histogram = px.histogram(dados_filt2, x="preco", nbins=15, text_auto=True).update_layout(yaxis_title="Quantidade dias") 
st.plotly_chart(histogram)


#----------------------
# Volatilidade 30 dias
#----------------------
st.write("### **Volatilidade do preço (30 dias)**")
dados_filt2['Volatilidade'] = dados_filt2['preco'].rolling(window=30).std()
vol = px.line(dados_filt2, x="data", y="Volatilidade", title="Volatilidade (30 dias) do Preço do Petróleo ao Longo do Tempo")
st.plotly_chart(vol)


#----------------------
# Decomposição da série temporal
#----------------------
st.write("### **Decomposicao da série temporal**")
result = seasonal_decompose(dados_filt2['preco'], model='multiplicative', period=30)
result = result.plot()
st.pyplot(result)