import streamlit as st
import pandas as pd
import numpy as np
import dataframe


iris_df = dataframe.df
iris_target = dataframe.target
model = dataframe.gnb


def select_sepal_length():
    sepal_length = st.sidebar.slider("Sepal Length:", 
               np.min(iris_df['sepal length (cm)']),
               np.max(iris_df['sepal length (cm)'])
        	   )
    return sepal_length

def select_sepal_width():
    sepal_width = st.sidebar.slider("Sepal Width:", 
               np.min(iris_df['sepal width (cm)']),
               np.max(iris_df['sepal width (cm)'])
        	   )
    return sepal_width

def select_petal_length():
    petal_length = st.sidebar.slider("Petal Length:", 
               np.min(iris_df['petal length (cm)']),
               np.max(iris_df['petal length (cm)'])
        	   )
    return petal_length

def select_petal_width():
    petal_width = st.sidebar.slider("Petal Width:", 
               np.min(iris_df['petal width (cm)']),
               np.max(iris_df['petal width (cm)'])
        	   )
    return petal_width

def get_predicted_value(sepal_length, sepal_width, petal_length, petal_width):
    return model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))

def app():
    sepal_length = select_sepal_length()
    sepal_width = select_sepal_width()
    petal_length = select_petal_length()
    petal_width = select_petal_width()

    prediction = get_predicted_value(sepal_length, sepal_width, petal_length, petal_width)

    st.write(
        iris_df.head())

    st.write('Predição:', prediction[0])

    