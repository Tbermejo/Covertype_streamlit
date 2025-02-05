import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Configuración de la página
st.set_page_config(page_title="Coberturas forestales", layout="wide")

def main():
    st.title("🖼️ Bosque Nacional Roosevelt del norte de Colorado")
    st.write("""Clasificación de píxeles en 7 tipos de cobertura forestal según atributos como elevación, aspecto, pendiente, sombreado, tipo de suelo.""")

    # Cargar el dataset covertype
    covertype = fetch_ucirepo(id=31)
    X = covertype.data.features
    y = covertype.data.targets
    dataset = pd.concat([X, y], axis=1)
    dataset.columns = list(X.columns) + ['target']
    dataset["target"] = dataset["target"].astype(str)  # Convertir la variable objetivo en categórica

    # Sidebar
    st.sidebar.header("Opciones de visualización")
    
    # Mostrar primeras filas del dataset
    if st.sidebar.checkbox("Mostrar primeras filas del dataset"):
        n_rows = st.sidebar.slider("Número de filas a mostrar:", 1, len(dataset), 5)
        st.write(f"### Primeras {n_rows} filas del dataset")
        st.write(dataset.head(n_rows))
    
    # Mostrar información general
    if st.sidebar.checkbox("Mostrar información general"):
        st.write("### Información general del dataset")
        st.write("#### Tipos de datos y valores nulos:")
        st.write(dataset.dtypes)
        st.write("#### Valores nulos por columna:")
        st.write(dataset.isnull().sum())
        st.write("#### Estadísticas descriptivas:")
        st.write(dataset.describe())
    
    # Selección de gráficos
    st.sidebar.header("Visualización de gráficos")
    chart_type = st.sidebar.selectbox(
        "Selecciona el tipo de gráfico:",
        [
            "Dispersión", "Histograma", "Boxplot", "Matriz de dispersión",
            "Mapa de correlación", "Gráfico de densidad (KDE)", "Treemap"
        ]
    )

    numeric_columns = dataset.select_dtypes(include=["float64", "int64"]).columns
    categorical_columns = dataset.select_dtypes(include=["object", "category"]).columns

    if chart_type == "Dispersión" and len(numeric_columns) > 1:
        x_var = st.sidebar.selectbox("Variable X:", numeric_columns)
        y_var = st.sidebar.selectbox("Variable Y:", numeric_columns)
        st.write(f"### Gráfico de dispersión: {x_var} vs {y_var}")
        fig = px.scatter(dataset, x=x_var, y=y_var, title=f"Dispersión de {x_var} vs {y_var}")
        st.plotly_chart(fig)
    
    elif chart_type == "Histograma" and len(numeric_columns) > 0:
        x_var = st.sidebar.selectbox("Variable para el histograma:", numeric_columns)
        st.write(f"### Histograma de {x_var}")
        fig = px.histogram(dataset, x=x_var, nbins=30, title=f"Distribución de {x_var}")
        st.plotly_chart(fig)
    
    elif chart_type == "Boxplot" and len(numeric_columns) > 0:
        y_var = st.sidebar.selectbox("Variable para el Boxplot:", numeric_columns)
        st.write(f"### Boxplot de {y_var}")
        fig = px.box(dataset, y=y_var, title=f"Boxplot de {y_var}")
        st.plotly_chart(fig)
    
    elif chart_type == "Matriz de dispersión" and len(numeric_columns) > 1:
        st.write("### Matriz de dispersión")
        fig = px.scatter_matrix(dataset, dimensions=numeric_columns, title="Matriz de dispersión")
        st.plotly_chart(fig)
    
    elif chart_type == "Mapa de correlación" and len(numeric_columns) > 1:
        st.write("### Mapa de correlación")
        corr_matrix = dataset[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", cbar=True, fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)
    
    elif chart_type == "Gráfico de densidad (KDE)" and len(numeric_columns) > 0:
        x_var = st.sidebar.selectbox("Variable para el gráfico de densidad:", numeric_columns)
        st.write(f"### Gráfico de densidad de {x_var}")
        fig = px.density_contour(dataset, x=x_var, title=f"Gráfico de densidad de {x_var}")
        st.plotly_chart(fig)
    
    elif chart_type == "Treemap" and len(categorical_columns) > 0:
        cat_var = st.sidebar.selectbox("Variable categórica para el Treemap:", categorical_columns)
        st.write("### Treemap basado en categorías")
        fig = px.treemap(dataset, path=[cat_var], values=numeric_columns[0], title="Treemap de categorías y valores")
        st.plotly_chart(fig)
    else:
        st.sidebar.warning("No hay suficientes variables disponibles para este gráfico.")

if __name__ == "__main__":
    main()

