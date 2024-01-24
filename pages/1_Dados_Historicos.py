
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


##----------------------------------------------------------------------------------
# Carregando dados e introducao
##----------------------------------------------------------------------------------
# Importando dados 
dados = pd.read_csv("./dados/historico_petroleo.csv")
dados["data"] = pd.to_datetime(dados["data"])
dados["preco"] = dados["preco_petroleo_bruto_brent"]

# Config iniciais
st.set_page_config(page_title="Dados histórico", page_icon="📈")
st.write("# Visualize aqui os dados históricos do preço do petróleo!")
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
st.markdown(
'''
### 
### 
### Ou então: selecione a data exata que deseja filtrar
'''
)
start_date = st.date_input("Selecione a data inicio", min(dados["data"]))
end_date = st.date_input("Selecione a data fim", max(dados["data"]))
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
dados_filt = dados[(dados["data"] >= start_date) & (dados["data"] <= end_date)]


chart = alt.Chart(dados_filt).mark_line().encode(
    x='data:T',
    y='preco:Q',
).properties(
    width=1000,
    height=600
)
# Exibindo o gráfico no Streamlit
st.altair_chart(chart, use_container_width=True)


#start_date = pd.to_datetime('1987-07-01')
#end_date = pd.to_datetime('1999-12-31')
#dados_filt = dados[(dados["data"] >= start_date) & (dados["data"] <= end_date)]