import streamlit as st
import main
import naive_bayes
import logistic_regression
import svm
import decision_tree
from multipage import MultiPage

app = MultiPage()

## titulo principal
st.title("Sklearn e Streamlit - Tópicos Especiais em Inteligência Artificial")

app.add_page("Página Inicial", main.app)
app.add_page("Naive Bayes", naive_bayes.app)
app.add_page("Logistic Regression", logistic_regression.app)
app.add_page("Support Vector Classifier", svm.app)
app.add_page("Decision Tree", decision_tree.app)


app.run()