# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:47:47 2024

@author: migue y JC
"""

# Librerías
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import plotly.express as px
from sklearn.datasets import load_iris
#from sklearn.metrics import confusion_matrix
#%%

# Base de datos de iris, nos servirá de default
iris = load_iris(as_frame=True)
df_default = iris['data']
df_default['species'] = iris['target']
df_default['species'] = df_default['species'].apply(lambda x: iris['target_names'][x])

#%%

# Título de la aplicación
st.title("Aplicación de Visualización de Datos")

# Cargar Dataset
uploaded_file = st.file_uploader("Carga tu dataset (CSV o Excel)", type=["csv", "xlsx"])

# Variable para almacenar el dataframe
df = None

if uploaded_file is not None:
    # Leer el dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

else:
    # Usar el dataset de iris como ejemplo si no se carga otro dataset
    df = df_default
    st.warning("No se ha subido un dataset. Usando dataset por defecto (Iris).")

# Mostrar las primeras filas del dataset
st.write("Primeras filas del dataset:")
st.dataframe(df.head(10))

# Análisis básico
st.write("Estadísticas descriptivas:")
st.write(df.describe())

# Preguntar al usuario si desea hacer un gráfico
if st.checkbox("¿Deseas generar un gráfico?"):
    # Seleccionar el tipo de gráfico
    chart_type = st.selectbox("Selecciona el tipo de gráfico", 
                              ["Gráfico de dispersión", "Histograma", "Gráfico de barras", "Gráfico de pastel", "Gráfico de caja"])

    # Opciones para cada tipo de gráfico
    if chart_type == "Gráfico de dispersión":
        x_axis = st.selectbox("Selecciona la columna para el eje X", df.columns)
        y_axis = st.selectbox("Selecciona la columna para el eje Y", df.columns)
        
        # Crear gráfico de dispersión usando seaborn
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
        ax.set_title(f"Gráfico de dispersión: {x_axis} vs {y_axis}")
        st.pyplot(fig)

    elif chart_type == "Histograma":
        column_for_histogram = st.selectbox("Selecciona la columna para el histograma", df.columns)

        # Crear histograma usando seaborn
        fig, ax = plt.subplots()
        sns.histplot(df[column_for_histogram], kde=True, ax=ax)
        ax.set_title(f"Histograma de {column_for_histogram}")
        st.pyplot(fig)

    elif chart_type == "Gráfico de barras":
        x_axis = st.selectbox("Selecciona la columna categórica para el eje X", df.columns)
        y_axis = st.selectbox("Selecciona la columna numérica para el eje Y", df.columns)

        # Crear gráfico de barras usando seaborn
        fig, ax = plt.subplots()
        sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)
        ax.set_title(f"Gráfico de barras: {x_axis} vs {y_axis}")
        st.pyplot(fig)

    elif chart_type == "Gráfico de pastel":
        # Gráfico de pastel no es directamente soportado por seaborn, se puede usar matplotlib
        column_for_pie = st.selectbox("Selecciona la columna para el gráfico de pastel", df.columns)

        fig, ax = plt.subplots()
        df[column_for_pie].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel('')
        ax.set_title(f"Gráfico de pastel de {column_for_pie}")
        st.pyplot(fig)

    elif chart_type == "Gráfico de caja":
        # Seleccionar columnas para los ejes X e Y
        x_axis = st.selectbox("Selecciona la columna categórica para el eje X", df.columns)
        y_axis = st.selectbox("Selecciona la columna numérica para el eje Y", df.columns)

        # Crear gráfico de caja usando seaborn
        fig, ax = plt.subplots()
        sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax)
        ax.set_title(f"Gráfico de caja: {x_axis} vs {y_axis}")
        st.pyplot(fig)

# Mapa de Calor
    if st.checkbox("¿Deseas generar un mapa de correlación?"):
        # Seleccionar solo las columnas numéricas
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if len(numeric_columns) > 1:
            corr = df[numeric_columns].corr()
            
            # Crear un mapa de correlación
            plt.figure(figsize=(12,8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
            plt.title("Matriz de Correlación")
            st.pyplot(plt)
    else:
        st.warning("No se pudo realizar el mapa de calor.")
#%%
