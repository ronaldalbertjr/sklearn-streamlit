import streamlit as st
import pandas as pd
import numpy as np
import dataframe
import plotly.graph_objects as go

iris_df = dataframe.df
iris_target = dataframe.target_gnb
model = dataframe.log


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

def generate_clf_graph(petal_length, petal_width, sepal_length, pred):
    fig = go.Figure(data = go.Scatter3d(x=iris_df['petal length (cm)'].values, 
                                        y=iris_df['petal width (cm)'].values, 
                                        z=iris_df['sepal length (cm)'].values,
                                        mode='markers',
                                        marker=dict(color=iris_target)))

    fig.add_trace(
            go.Scatter3d(x=np.array([petal_length]),
                       y=np.array([petal_width]),
                       z=np.array([sepal_length]),
                       mode='markers',
            )
    )

    return fig

def app():
    sepal_length = select_sepal_length()
    sepal_width = select_sepal_width()
    petal_length = select_petal_length()
    petal_width = select_petal_width()

    prediction = get_predicted_value(sepal_length, sepal_width, petal_length, petal_width)

    st.write(
        iris_df.head())

    st.write('Predição:', prediction[0])

    clf_graph = generate_clf_graph(petal_length, petal_width, sepal_length, prediction[0])
    st.plotly_chart(clf_graph)

    st.write('Matriz de Confusão:', dataframe.gnb_matrix)