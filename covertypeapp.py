import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Configuración de la página
st.set_page_config(page_title="Dataset Forest Covertype", layout="wide")

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

st.title("Métodos de clasificación para la predicción de coberturas forestales")

if capitulo == "Introducción":
    st.write("""El dataset Covertype proporciona información de cuatro áreas naturales localizadas en el Parque Natural Roosevelt en el Norte de Colorado, Estados Unidos.
    El objetivo es clasificar el tipo de cobertura forestal según variables cartográficas como: """)

# Definir los datos de las variables en un DataFrame
    variables_info = pd.DataFrame({
        "Variable": [
            "Elevación", "Orientación", "Pendiente", "Distancia_horizontal_a_hidrología",
            "Distancia_vertical_a_hidrología", "Distancia_horizontal_a_carreteras", 
            "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", 
            "Horizontal_Distance_To_Fire_Point", "Área silvestre 1", 
            "Área silvestre 2", "Área silvestre 3", "Área silvestre 4", 
            "Tipo de suelo 1-40"
        ],
        "Descripción": [
            "Elevación en metros.",
            "Orientación en grados de acimut.",
            "Pendiente en grados.",
            "Distancia horizontal a las características de agua superficial más cercanas.",
            "Distancia vertical a las características de agua superficial más cercanas.",
            "Distancia horizontal a la carretera más cercana.",
            "Índice de sombra de las colinas a las 9 a. m., solsticio de verano. Valor de 255.",
            "Índice de sombra de las colinas al mediodía, solsticio de verano. Valor de 255.",
            "Índice de sombra de las colinas a las 3 p. m., solsticio de verano. Valor de 255.",
            "Distancia horizontal a los puntos de ignición de incendios forestales más cercanos.",
            "Área silvestre Rawah.",
            "Área silvestre Neota.",
            "Área silvestre Comanche Peak.",
            "Área silvestre Cache la Poudre.",
            "Tipos de suelo categorizados del 1 al 40."
        ]
    })
    
    st.write("### 📋 Variables del Dataset")
    st.table(variables_info)
    
# Variable objetivo
    st.write("""Donde la variable objetivo es el tipo de cobertura forestal, descrita a continuación:""")

    variable_obj = pd.DataFrame({
        "Tipo de cobertura": [
            "Spruce/Fir - Pícea/abeto","Lodgepole Pine - Pino contorta","Ponderosa Pine - Pino ponderosa",
            "Cottonwood/Willow - Álamo de Virginia/sauce","Aspen - Álamo temblón","Douglas-fir - Abeto de Douglas","Krummholz"
        ],

        "ID": [
            "1","2","3","4","5","6","7"
        ]
    })

    st.write("### 📋 Tipo de coberturas - Variable objetivo")
    st.table(variable_obj)
    
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
        ["Dispersión", "Distribución variable objetivo", "Matriz de dispersión",
         "Mapa de correlación"]
    )

    if chart_type == "Dispersión" and len(numeric_columns) > 1:
        x_var = st.sidebar.selectbox("Variable X:", numeric_columns)
        y_var = st.sidebar.selectbox("Variable Y:", numeric_columns)
        st.write(f"### Gráfico de dispersión: {x_var} vs {y_var}")
        fig = px.scatter(dataset, x=x_var, y=y_var, title=f"Dispersión de {x_var} vs {y_var}")
        st.plotly_chart(fig)
        
    elif chart_type == "Distribución Variable objetivo":
        st.write("### Distribución de la variable objetivo (Cover_Type)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=dataset, x='target', palette='viridis', ax=ax)
        ax.set_title("Distribución de la variable objetivo (Cover_Type)")
        ax.set_xlabel("Tipo de cobertura")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)
    
    elif chart_type == "Mapa de correlación" and len(numeric_columns) > 1:
        st.write("### Mapa de correlación")
        corr = dataset.corr()
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Mapa de correlación")
        st.pyplot(fig)
    
elif capitulo == "Modelos de Clasificación":
    st.header("🤖 Modelos de Clasificación")
    st.write("Aquí se implementarán y compararán diferentes modelos de clasificación.")

    # Espacio para incluir la implementación de modelos más adelante


