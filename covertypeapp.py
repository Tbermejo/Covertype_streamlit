import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Configuración de la página
st.set_page_config(page_title="Coberturas forestales", layout="wide")

def cargar_datos():
    """Carga el dataset covertype y lo formatea."""
    covertype = fetch_ucirepo(id=31)
    X = covertype.data.features
    y = covertype.data.targets
    dataset = pd.concat([X, y], axis=1)
    dataset.columns = list(X.columns) + ['target']
    dataset["target"] = dataset["target"].astype(str)
    return dataset

# Cargar el dataset
dataset = cargar_datos()
numeric_columns = dataset.select_dtypes(include=["float64", "int64"]).columns
categorical_columns = dataset.select_dtypes(include=["object", "category"]).columns

# Barra lateral: Selección de capítulos
st.sidebar.title("📚 Capítulos")
capitulo = st.sidebar.radio("Selecciona un capítulo:", [
    "Introducción",
    "Exploración de Datos",
    "Visualización de Datos",
    "Modelos de Clasificación"
])

st.title("🖼️ Bosque Nacional Roosevelt del norte de Colorado")

if capitulo == "Introducción":
    st.write("""Este estudio busca clasificar píxeles en 7 tipos de cobertura forestal 
    según atributos como elevación, aspecto, pendiente, sombreado y tipo de suelo.""")

elif capitulo == "Exploración de Datos":
    st.header("🔍 Exploración de Datos")

    if st.sidebar.checkbox("Mostrar primeras filas"):
        n_rows = st.sidebar.slider("Número de filas a mostrar:", 1, len(dataset), 5)
        st.write(f"### Primeras {n_rows} filas del dataset")
        st.write(dataset.head(n_rows))
    
    if st.sidebar.checkbox("Mostrar información general"):
        st.write("### Información general del dataset")
        st.write("#### Tipos de datos y valores nulos:")
        st.write(dataset.dtypes)
        st.write("#### Valores nulos por columna:")
        st.write(dataset.isnull().sum())
        st.write("#### Estadísticas descriptivas:")
        st.write(dataset.describe())

elif capitulo == "Visualización de Datos":
    st.header("📊 Visualización de Datos")

    chart_type = st.sidebar.selectbox(
        "Selecciona el tipo de gráfico:",
        ["Dispersión", "Histograma", "Boxplot", "Matriz de dispersión",
         "Mapa de correlación", "Gráfico de densidad (KDE)", "Treemap"]
    )

    if chart_type == "Dispersión" and len(numeric_columns) > 1:
        x_var = st.sidebar.selectbox("Variable X:", numeric_columns)
        y_var = st.sidebar.selectbox("Variable Y:", numeric_columns)
        st.write(f"### Gráfico de dispersión: {x_var} vs {y_var}")
        fig = px.scatter(dataset, x=x_var, y=y_var, title=f"Dispersión de {x_var} vs {y_var}")
        st.plotly_chart(fig)
    
    elif chart_type == "Mapa de correlación" and len(numeric_columns) > 1:
        st.write("### Mapa de correlación")
        corr_matrix = dataset[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", cbar=True, fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

elif capitulo == "Modelos de Clasificación":
    st.header("🤖 Modelos de Clasificación")
    st.write("Aquí se implementarán y compararán diferentes modelos de clasificación.")

    # Espacio para incluir la implementación de modelos más adelante


