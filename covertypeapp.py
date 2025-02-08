import streamlit as st
import pandas as pd
import numpy as np
import gzip
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

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
    dataset["target"] = dataset["target"].apply(lambda x: x if x in [1, 2] else 3)
    dataset["target"] = dataset["target"].astype(str)
    return dataset

# Cargar el dataset
dataset = cargar_datos()
X = dataset.drop(columns=["target"])  # Variables predictoras
y = dataset["target"]  # Variable objetivo

# Ver distribución de clases
st.write("📊 **Distribución de clases después de reclasificación:**")
st.write(y.value_counts())


# Mostrar información en Streamlit
st.write(f"✅ **Datos cargados con {dataset.shape[0]} filas y {dataset.shape[1]} columnas.**")
st.write("🔹 **Ejemplo de datos:**", dataset.head())

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
# Diccionario con nombres de modelos y sus rutas
model_paths = {
    "Modelo K Nearest Neighbors": "best_model_trained_classifier_new.pkl.gz",
    "Modelo Red Neuronal": "model_trained_neuronal.pkl.gz",
    
}

# Sidebar para elegir el modelo
modelo_seleccionado = st.sidebar.selectbox("Seleccione el modelo de clasificación", list(model_paths.keys()))

# Cargar el modelo seleccionado
@st.cache_resource
def cargar_modelo(ruta):
    with gzip.open(ruta, "rb") as file:
        return pickle.load(file)

modelo = cargar_modelo(model_paths[modelo_seleccionado])

st.title("Métodos de clasificación para el Dataset Covertype")

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
            "Orientación en grados de azimut.",
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
            "Spruce/Fir - Pícea/abeto","Lodgepole Pine - Pino contorta","Otras"
    
        ],

        "ID": [
            "1","2","3"
        ]
    })

    st.write("### 📋 Tipo de coberturas - Variable objetivo")
    st.table(variable_obj)

    st.write("📊 **Distribución de clases después de reclasificación:**")
    st.write(y.value_counts())
    st.write("""Fuente: Blackard, J. (1998). Covertype [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N.""")
elif capitulo == "Exploración de Datos":
    st.header("🔍 Exploración de Datos")

    if st.checkbox("Mostrar primeras filas"):
        n_rows = st.slider("Número de filas a mostrar:", 1, len(dataset), 5)
        st.write(f"### Primeras {n_rows} filas del dataset")
        st.write(dataset.head(n_rows))
    
    if st.checkbox("Mostrar información general"):
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
        ["Dispersión", "Distribución variable objetivo",
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
    st.header("🤖 K- Nearest Neighbors")
    st.write("Información del modelo previamente entrenado por el método K Nearest Neighbors.")

    
    #Información del modelo
    st.write("📊 Parámetros del Modelo")
    modelo = cargar_modelo(model_paths[modelo_seleccionado])
    if modelo is not None:
        modelo_tipo = type(modelo).__name__
        st.write(f"📌 **Tipo de modelo:** {modelo_tipo}")

        # Si el modelo es un pipeline, extraer la última etapa (el modelo real)
        if isinstance(modelo, Pipeline):
            modelo_real = modelo.named_steps.get("reg")  # Extrae la etapa "reg" (el regresor)
            scaler_usado = modelo.named_steps.get("scaler")
        else:
            modelo_real = modelo
            scaler_usado = None

        if modelo_real:
            st.write(f"🛠 **Modelo en uso:** {type(modelo_real).__name__}")
    
        if scaler_usado:
            st.write(f"⚖️ **Escalador aplicado:** {type(scaler_usado).__name__}")

        try:
            if model is not None:
                params = model.get_params()
            else:
                params = {}
                st.write("### 🔧 Hiperparámetros Ajustados:")

        # Diccionario con descripciones de los hiperparámetros más comunes
            hyperparam_descriptions = {
                "model__n_neighbors": "Número óptimo de vecinos más cercanos.",
                "model__p": "métrica de distancia utilizada para calcular la similitud entre puntos."
            }

            for key, value in params.items():
                explanation = hyperparam_descriptions.get(key, "Sin descripción disponible")
                st.write(f"🔹 **{key}:** {value}")
                st.caption(f"📘 {explanation}")  

    
        except Exception as e:
            st.error(f"⚠️ Error al obtener los hiperparámetros del modelo: {e}")


            # Mostrar precisión en la barra lateral de Streamlit
        st.write("### 📏 Precisión del modelo:")
        
#####Aquí

    st.header("🤖 Modelo Redes Neuronales")
    st.write("Información del modelo previamente entrenado por el método redes neuronales.")

    # Definir las características que necesita el modelo
feature_names = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Horizontal_Distance_To_Fire_Point"
]

#Rango de valores para las variables
variables_range = {
    "Elevation": {"min": 1850, "max": 4000, "desc": "Elevación en metros"},
    "Aspect": {"min": 0, "max": 360, "desc": "Orientación en grados de azimut"},
    "Slope": {"min": 0, "max": 60, "desc": "Pendiente en grados"},
    "Horizontal_Distance_To_Hydrology": {"min": 0, "max": 1350, "desc": "Distancia a cuerpos de agua"},
    "Vertical_Distance_To_Hydrology": {"min": -150, "max": 550, "desc": "Diferencia de altura cuerpos de agua"},
    "Horizontal_Distance_To_Roadways": {"min": 0, "max": 7000, "desc": "Distancia a la carretera"},
    "Horizontal_Distance_To_Fire_Point": {"min": 0, "max": 7000, "desc": "Distancia a punto de incendios"}
}

#Ingresar variables para clasificación
st.sidebar.header("📌 Ingrese los valores para clasificar el tipo de cobertura:")

valores_usuario = []
for col, info in variables_range.items():
    valor = st.sidebar.slider(
        f"{col} - {info['desc']}",
        min_value=float(info["min"]),
        max_value=float(info["max"]),
        value=(info["min"] + info["max"]) / 2
    )
    valores_usuario.append(valor)

# Botón de clasificación
if st.sidebar.button("🔍 Clasificar Cobertura"):
    if modelo is not None:
        entrada = np.array(valores_usuario).reshape(1, -1)  # Convertir a matriz

        # Verificar si el modelo es una red neuronal
        if hasattr(modelo, "predict_proba"):  
            entrada = entrada.astype(np.float32)  # Convertir a float32 si es necesario

        try:
            prediccion = modelo.predict(entrada)  # Hacer la predicción

            # Si la predicción es un array de probabilidades, convertir a clase
            if len(prediccion.shape) > 1 and prediccion.shape[1] > 1:
                prediccion = np.argmax(prediccion, axis=1)  # Tomar la clase con mayor probabilidad

            st.sidebar.success(f"🌲 Tipo de cobertura clasificada: {int(prediccion[0])}")  
        except Exception as e:
            st.error(f"⚠️ Error al hacer la predicción: {e}")
    else:
        st.error("⚠️ No se pudo hacer la clasificación porque el modelo no está cargado.")



