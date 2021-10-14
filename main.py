from scipy.sparse import data
import streamlit as st
import dataframe

def app():

    mi_df = dataframe.mi_df
    
    
    st.markdown("""Trabalho Final da Disciplina de Tópicos Especiais em Inteligência Artificial.<br>
                O projeto contém 3 páginas, cada uma associada à um modelo de Aprendizado Estátístico, para classificar
                o tipo de flor íris.
                """, unsafe_allow_html=True
    )

    st.markdown("""Abaixo podemos ver a tabela associando a Informação Mútua de cada uma das <i>features</i> com a classificação
                do tipos de iris.<br>
                Usaremos a seguinte tabela para construir nossos gráficos, uma vez que só podemos construir gráficos em 3 dimensões,
                escolheremos as 3 dimensões que possuem maior Informação Mútua com o <i>target</i>.
                """, unsafe_allow_html=True
    )
    st.write(mi_df)
