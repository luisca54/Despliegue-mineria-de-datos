import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================
# CONFIGURACIÓN DE PÁGINA
# ==========================

st.set_page_config(
    page_title="Monitoreo Sistema Hidráulico",
    page_icon="⚙️",
    layout="wide"
)

# ==========================
# CARGA DE ARCHIVOS
# ==========================

modelo = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")
columnas = joblib.load("columnas.pkl")
medias = joblib.load("medias.pkl")

importance = pd.read_csv(
    "feature_importance.csv"
)

# ==========================
# TÍTULO
# ==========================

st.title("⚙️ Sistema de Monitoreo Hidráulico")

st.write(
    """
    Aplicación desarrollada para predecir la estabilidad de un sistema hidráulico
    utilizando un modelo Random Forest entrenado con datos de sensores industriales.
    """
)

# ==========================
# MÉTRICAS DEL MODELO
# ==========================

st.subheader("Desempeño del Modelo")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        "98.49%"
    )

with col2:
    st.metric(
        "F1 Score",
        "0.98"
    )

with col3:
    st.metric(
        "AUC",
        "0.9966"
    )

# ==========================
# COMPARACIÓN DE MODELOS
# ==========================

st.subheader("Comparación de Modelos")

resultados = pd.DataFrame({
    "Modelo": [
        "Random Forest",
        "MLP",
        "KNN"
    ],
    "Accuracy": [
        98.49,
        97.58,
        96.07
    ]
})

st.bar_chart(
    resultados.set_index("Modelo")
)

# ==========================
# IMPORTANCIA VARIABLES
# ==========================

st.subheader("Variables Más Importantes")

top10 = importance.head(10)


fig, ax = plt.subplots(figsize=(4, 2.5))

ax.barh(
    top10["Variable"],
    top10["Importancia"]
)

ax.set_xlabel("Importancia")
ax.set_ylabel("Variable")
ax.invert_yaxis()

st.pyplot(fig, use_container_width=False)


# ==========================
# PREDICCIÓN
# ==========================

st.header("Predicción")

modo = st.radio(
    "Seleccione el método de entrada",
    [
        "Manual",
        "Aleatorio",
        "CSV"
    ]
)

# =====================================
# MODO MANUAL
# =====================================

if modo == "Manual":

    st.subheader("Ingreso Manual")

    datos_usuario = {}

    col_1, col_2 = st.columns(2)

    mitad = len(columnas) // 2

    columnas_1 = columnas[:mitad]
    columnas_2 = columnas[mitad:]

    with col_1:
        for col in columnas_1:
            datos_usuario[col] = st.number_input(
                col,
                value=float(medias[col]),
                format="%.4f"
            )

    with col_2:
        for col in columnas_2:
            datos_usuario[col] = st.number_input(
                col,
                value=float(medias[col]),
                format="%.4f"
            )

    if st.button("Predecir"):

        entrada = pd.DataFrame([datos_usuario])

        entrada_scaled = scaler.transform(
            entrada
        )

        pred = modelo.predict(
            entrada_scaled
        )

        proba = modelo.predict_proba(
            entrada_scaled
        )

        if pred[0] == 0:

            st.success(
                "✅ Sistema Estable"
            )

        else:

            st.error(
                "⚠️ Sistema Potencialmente Inestable"
            )

        st.write(
            f"Probabilidad Estable: {proba[0][0]*100:.2f}%"
        )

        st.write(
            f"Probabilidad Inestable: {proba[0][1]*100:.2f}%"
        )

# =====================================
# MODO ALEATORIO
# =====================================

elif modo == "Aleatorio":

    st.subheader("Generar Registro Aleatorio")

    if st.button("Generar Datos"):

        fila = {}

        for col in columnas:

            media = float(medias[col])

            fila[col] = np.random.normal(
                loc=media,
                scale=abs(media * 0.05)
            )

        entrada = pd.DataFrame([fila])

        st.write("Datos generados:")

        st.dataframe(entrada)

        entrada_scaled = scaler.transform(
            entrada
        )

        pred = modelo.predict(
            entrada_scaled
        )

        proba = modelo.predict_proba(
            entrada_scaled
        )

        if pred[0] == 0:

            st.success(
                "✅ Sistema Estable"
            )

        else:

            st.error(
                "⚠️ Sistema Potencialmente Inestable"
            )

        st.write(
            f"Probabilidad Estable: {proba[0][0]*100:.2f}%"
        )

        st.write(
            f"Probabilidad Inestable: {proba[0][1]*100:.2f}%"
        )

# =====================================
# MODO CSV
# =====================================

elif modo == "CSV":

    st.subheader("Carga de Archivo CSV")

    uploaded_file = st.file_uploader(
        "Seleccione un archivo CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        datos = pd.read_csv(
            uploaded_file
        )

        datos_scaled = scaler.transform(
            datos
        )

        predicciones = modelo.predict(
            datos_scaled
        )

        probabilidades = modelo.predict_proba(
            datos_scaled
        )

        resultado = datos.copy()

        resultado["Prediccion"] = predicciones

        resultado["Prob_Estable"] = (
            probabilidades[:, 0] * 100
        )

        resultado["Prob_Inestable"] = (
            probabilidades[:, 1] * 100
        )

        st.dataframe(resultado)

        csv = resultado.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Descargar resultados",
            data=csv,
            file_name="predicciones.csv",
            mime="text/csv"
        )

# ==========================
# FOOTER
# ==========================

st.markdown("---")

st.caption(
    "Proyecto de Minería de Datos - Monitoreo de Sistemas Hidráulicos"
)
