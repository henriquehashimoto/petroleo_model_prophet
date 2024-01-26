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
    - Dados utilizados
        - Visualização histórica dos preços
        - Análise Exploratória de Dados
    - Previsão do preço diário utilizando modelos de ML  
    
    ### Integrantes:
    - Bruno Silva Lopes
    - Henrique Eiji Hashimoto
    - Rodrigo Araújo
    - Roney Cliento Molina
    
    ### Dash e Storytelling:
    - Clique [aqui](https://lookerstudio.google.com/u/0/reporting/b08ea7d4-1cc4-4cfe-878e-c35190f74b45/page/Sw2nD) para acessar o dashboard e entenda o porque das grandes variações de preços do petróleo
"""
)
