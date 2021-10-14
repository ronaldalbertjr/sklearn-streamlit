import streamlit as st
import main
import naive_bayes
import logistic_regression
from multipage import MultiPage

app = MultiPage()

## titulo principal
st.title("Projeto Final - Tópicos Especiais em Inteligência Artificial")

app.add_page("Página Inicial", main.app)
app.add_page("Naive Bayes", naive_bayes.app)
app.add_page("Logistic Regression", logistic_regression.app)

app.run()