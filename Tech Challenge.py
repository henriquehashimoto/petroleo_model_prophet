import streamlit as st

st.set_page_config(
    page_title="Bem vindo(a)!",
    page_icon=":bar_chart:",
)

st.write("# Seja muito bem vindo(a) ao nosso Tech Challenge 4! :grin:")

st.sidebar.success("Selecione uma das páginas acima")

st.markdown(
    """  
    ### Aqui você encontra:
    - Dados histórico do preço do barril do petróleo Brent
    - Previsão do preço diário utilizando modelos de ML  
    
    ### Integrantes:
    - Bruno Silva Lopes
    - Henrique Eiji Hashimoto
    - Rodrigo Araújo
    - Roney Cliento Molina
"""
)