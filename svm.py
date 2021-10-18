import streamlit as st
import pandas as pd
import numpy as np
import dataframe
import plotly.graph_objects as go


iris_df = dataframe.df

def select_kernel_type():
    kernel = st.sidebar.selectbox("Escolha o tipo do kernel no SVM:", 
                                    dataframe.kernel_types
                                    )
    return kernel

def get_model(kernel):
    model = dataframe.svc[kernel]

    return model

def get_confusion_matrix(kernel):
    confusion_matrix = dataframe.svc_matrix[kernel]
    
    return confusion_matrix

def get_target(kernel):
    iris_target = dataframe.svc_target[kernel]
    
    return iris_target

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

def get_predicted_value(sepal_length, sepal_width, petal_length, petal_width, model):
    return model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))

def generate_clf_graph(petal_length, petal_width, sepal_length, pred, target):
    fig = go.Figure(data = go.Scatter3d(x=iris_df['petal length (cm)'].values, 
                                        y=iris_df['petal width (cm)'].values, 
                                        z=iris_df['sepal length (cm)'].values,
                                        mode='markers',
                                        marker=dict(color=target)))

    fig.add_trace(
            go.Scatter3d(x=np.array([petal_length]),
                       y=np.array([petal_width]),
                       z=np.array([sepal_length]),
                       mode='markers',
            )
    )

    return fig

def app():
    
    kernel = select_kernel_type()

    iris_target = get_target(kernel)

    model = get_model(kernel)
    
    sepal_length = select_sepal_length()
    sepal_width = select_sepal_width()
    petal_length = select_petal_length()
    petal_width = select_petal_width()

    prediction = get_predicted_value(sepal_length, sepal_width, petal_length, petal_width, model)

    st.write(
        iris_df.head())

    st.write('Predição:', prediction[0])

    clf_graph = generate_clf_graph(petal_length, petal_width, sepal_length, prediction[0], iris_target)
    st.plotly_chart(clf_graph)
    
    st.write('Matriz de Confusão:', get_confusion_matrix(kernel))