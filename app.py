from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predicción de consumo de flota", page_icon="🚗", layout="centered")

MODEL_CANDIDATES = [
    Path("Grupo08_Modelo_Final.pkl"),
    Path("Grupo08_Modelo_Final.txt"),
]
UMBRAL_REVISION_PCT = 0.25
UMBRAL_REVISION_GAL = 5.0


@st.cache_resource
def cargar_paquete():
    ruta = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if ruta is None:
        raise FileNotFoundError(
            "No se encontró Grupo08_Modelo_Final.pkl ni Grupo08_Modelo_Final.txt. "
            "Sube el modelo al mismo repositorio que app.py."
        )
    paquete = joblib.load(ruta)
    if not isinstance(paquete, dict) or "modelo" not in paquete or "metadata" not in paquete:
        raise ValueError("El archivo no contiene las claves 'modelo' y 'metadata'.")
    return paquete, ruta.name


def crear_fila(carro, clase, mes_num, km_mes, galones_lag1, km_lag1, clases_interaccion):
    fila = pd.DataFrame([{
        "VEHICULO": carro,
        "CLASE": clase,
        "KM_MES": float(km_mes),
        "GALONES_LAG1": float(galones_lag1),
        "KM_MES_LAG1": float(km_lag1),
        "MES_SIN": np.sin(2 * np.pi * mes_num / 12),
        "MES_COS": np.cos(2 * np.pi * mes_num / 12),
    }])
    for clase_interaccion in clases_interaccion:
        nombre = "KM_X_" + "".join(
            caracter if caracter.isalnum() else "_"
            for caracter in str(clase_interaccion).upper()
        )
        fila[nombre] = float(km_mes) * (clase == clase_interaccion)
    return fila


def dinero_cop(valor):
    return "$" + f"{valor:,.0f}".replace(",", ".")


st.title("🚗 Predicción mensual de consumo")
st.caption("Estima los galones esperados de un vehículo y compara el resultado con el consumo facturado.")

try:
    paquete, archivo_modelo = cargar_paquete()
    modelo = paquete["modelo"]
    meta = paquete["metadata"]
except Exception as error:
    st.error(f"No fue posible cargar el modelo: {error}")
    st.stop()

vehiculos = pd.DataFrame(meta.get("vehiculos", []))
if vehiculos.empty or "VEHICULO" not in vehiculos.columns:
    st.error("El modelo no contiene la metadata de vehículos necesaria para la aplicación.")
    st.stop()

vehiculos = vehiculos.set_index("VEHICULO")
precio_galon = float(meta.get("precio_galon_referencia_cop", 0))
parametros = meta.get("parametros_externos", {})
factor_gasolina = float(parametros.get("factores", {}).get("gasolina_kg_co2_gal", 0))
proporcion_fosil = float(meta.get("proporcion_gasolina_fosil", 1.0))
if proporcion_fosil > 1:
    proporcion_fosil /= 100
co2_kg_por_galon = factor_gasolina * proporcion_fosil
clases_interaccion = meta.get("clases_interaccion", [])

with st.form("formulario_prediccion"):
    carro = st.selectbox("Vehículo", sorted(vehiculos.index.tolist()))
    datos_carro = vehiculos.loc[carro]
    clase = str(datos_carro.get("CLASE", "AUTOMOVIL"))

    col1, col2 = st.columns(2)
    with col1:
        periodo = st.text_input("Periodo", value="2026-01", help="Formato AAAA-MM")
        km_mes = st.number_input("Kilómetros del mes", min_value=1.0, max_value=10000.0, value=900.0, step=50.0)
    with col2:
        galones_facturados = st.number_input(
            "Galones facturados (opcional)", min_value=0.0, max_value=500.0, value=0.0, step=1.0
        )
        st.text_input("Clase del vehículo", value=clase.title(), disabled=True)

    usar_ultimo_mes = st.checkbox("Usar los datos anteriores guardados en el modelo", value=True)
    if usar_ultimo_mes:
        galones_lag1 = float(datos_carro.get("GALONES_ULTIMO_MES", 0) or 0)
        km_lag1 = float(datos_carro.get("KM_ULTIMO_MES", km_mes) or km_mes)
    else:
        col3, col4 = st.columns(2)
        with col3:
            galones_lag1 = st.number_input("Galones del mes anterior", min_value=0.0, value=30.0)
        with col4:
            km_lag1 = st.number_input("Kilómetros del mes anterior", min_value=0.0, value=900.0)

    predecir = st.form_submit_button("Predecir consumo", type="primary", use_container_width=True)

if predecir:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", periodo.strip()):
        st.error("El periodo debe tener el formato AAAA-MM, por ejemplo 2026-01.")
        st.stop()

    mes_num = int(periodo[-2:])
    fila = crear_fila(
        carro, clase, mes_num, km_mes, galones_lag1, km_lag1, clases_interaccion
    )

    try:
        galones_esperados = max(0.0, float(modelo.predict(fila)[0]))
    except Exception as error:
        st.error(f"No fue posible generar la predicción: {error}")
        st.stop()

    costo_esperado = galones_esperados * precio_galon
    co2_estimado = galones_esperados * co2_kg_por_galon

    c1, c2, c3 = st.columns(3)
    c1.metric("Galones esperados", f"{galones_esperados:,.1f}")
    c2.metric("Costo esperado", dinero_cop(costo_esperado))
    c3.metric("CO₂ estimado", f"{co2_estimado:,.0f} kg")

    resultado = {
        "Vehículo": carro,
        "Periodo": periodo,
        "Clase": clase,
        "Km del mes": km_mes,
        "Galones esperados": galones_esperados,
        "Costo esperado COP": costo_esperado,
        "CO2 estimado kg": co2_estimado,
    }

    if galones_facturados > 0:
        diferencia = galones_facturados - galones_esperados
        pct = diferencia / galones_esperados if galones_esperados > 0 else 0
        revisar = diferencia > UMBRAL_REVISION_GAL and pct > UMBRAL_REVISION_PCT

        resultado["Galones facturados"] = galones_facturados
        resultado["Diferencia"] = diferencia
        resultado["Estado"] = "Revisar" if revisar else "Normal"

        if revisar:
            st.error(
                f"⚠️ Revisar: el consumo facturado supera el esperado en {diferencia:,.1f} galones "
                f"({pct:.0%})."
            )
        else:
            st.success(f"✅ Consumo normal. Diferencia: {diferencia:,.1f} galones ({pct:.0%}).")
    else:
        resultado["Estado"] = "Predicción generada"
        st.info("No se ingresaron galones facturados, por lo que solo se presenta la predicción.")

    salida = pd.DataFrame([resultado])
    st.dataframe(salida, use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar resultado CSV",
        data=salida.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"prediccion_{carro}_{periodo}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with st.expander("Información del modelo"):
    st.write(f"**Archivo cargado:** {archivo_modelo}")
    st.write(f"**Modelo:** {meta.get('nombre_modelo', 'No disponible')}")
    st.write(f"**Versión de scikit-learn:** {meta.get('version_sklearn', 'No disponible')}")
    st.caption("La aplicación no reentrena el modelo. Solo utiliza el modelo ya entrenado para generar predicciones.")
