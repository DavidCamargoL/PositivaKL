from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Gestión de flota", page_icon="🚗", layout="wide")

MODEL_CANDIDATES = [Path("Grupo08_Modelo_Final.pkl"), Path("Grupo08_Modelo_Final.txt")]
UMBRAL_REVISION_PCT = 0.25
UMBRAL_REVISION_GAL = 5.0

CARROS_GASOLINA = {
    "AUTOMOVIL": {"Básico": ("Nissan Versa SENSE MT", 72_990_000), "Intermedio": ("Nissan Versa ADVANCE CVT", 82_990_000), "Full equipo": ("Nissan Versa EXCLUSIVE CVT", 88_990_000)},
    "CAMIONETA": {"Básico": ("Nissan Frontier S 4x2", 128_990_000), "Intermedio": ("Nissan Frontier PRO-S 4x2", 144_990_000), "Full equipo": ("Nissan Frontier PRO-4X 4x4", 179_990_000)},
}
CARROS_ELECTRICOS = {
    "Tesla Model Y": {"version": "Tesla Model Y Premium RWD", "precio": 119_990_000, "clases": ["AUTOMOVIL", "CAMIONETA"]},
    "BYD Yuan UP": {"version": "BYD Yuan UP 380 GS", "precio": 99_990_000, "clases": ["AUTOMOVIL", "CAMIONETA"]},
    "Tesla Model 3": {"version": "Tesla Model 3 Premium RWD", "precio": 114_990_000, "clases": ["AUTOMOVIL"]},
}

@st.cache_resource
def cargar_paquete():
    ruta = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if ruta is None:
        raise FileNotFoundError("No se encontró Grupo08_Modelo_Final.pkl en el repositorio.")
    paquete = joblib.load(ruta)
    return paquete, ruta.name

def crear_fila(carro, clase, mes_num, km_mes, galones_lag1, km_lag1, clases_interaccion):
    fila = pd.DataFrame([{"VEHICULO": carro, "CLASE": clase, "KM_MES": float(km_mes),
        "GALONES_LAG1": float(galones_lag1), "KM_MES_LAG1": float(km_lag1),
        "MES_SIN": np.sin(2*np.pi*mes_num/12), "MES_COS": np.cos(2*np.pi*mes_num/12)}])
    for c in clases_interaccion:
        nombre = "KM_X_" + "".join(x if x.isalnum() else "_" for x in str(c).upper())
        fila[nombre] = float(km_mes) * (clase == c)
    return fila

def dinero(v):
    return "$" + f"{v:,.0f}".replace(",", ".")

def millones(v):
    return f"${v/1_000_000:,.1f} M".replace(",", "X").replace(".", ",").replace("X", ".")

try:
    paquete, archivo_modelo = cargar_paquete()
    modelo, meta = paquete["modelo"], paquete["metadata"]
except Exception as e:
    st.error(f"No fue posible cargar el modelo: {e}")
    st.stop()

externos = meta.get("parametros_externos", {})
factores = externos.get("factores", {})
tarifas = externos.get("tarifas_cop_kwh", {"BOGOTA": 1750.0, "BARRANQUILLA": 1800.0})
vehiculos_ev = externos.get("vehiculos_electricos", {})
precio_galon_base = float(meta.get("precio_galon_referencia_cop", 0))
proporcion_fosil = float(meta.get("proporcion_gasolina_fosil", 1.0))
if proporcion_fosil > 1: proporcion_fosil /= 100
co2_galon = float(factores.get("gasolina_kg_co2_gal", 8.813)) * proporcion_fosil
co2_kwh = float(factores.get("electricidad_kg_co2_kwh", 0.097))
clases_interaccion = meta.get("clases_interaccion", [])
vehiculos = pd.DataFrame(meta.get("vehiculos", [])).set_index("VEHICULO")

st.title("🚗 Gestión y renovación de flota")
st.caption("Predicción de combustible y comparación básica entre vehículos de gasolina y eléctricos.")
tab1, tab2 = st.tabs(["⛽ Predicción de consumo", "⚡ Reemplazo de flota"])

with tab1:
    st.subheader("Predicción mensual")
    with st.form("prediccion"):
        carro = st.selectbox("Vehículo", sorted(vehiculos.index.tolist()))
        d = vehiculos.loc[carro]
        clase = str(d.get("CLASE", "AUTOMOVIL"))
        c1, c2, c3 = st.columns(3)
        periodo = c1.text_input("Periodo", "2026-01")
        km_mes = c2.number_input("Kilómetros del mes", 1.0, 10000.0, 900.0, 50.0)
        facturados = c3.number_input("Galones facturados (opcional)", 0.0, 500.0, 0.0, 1.0)
        usar = st.checkbox("Usar datos del mes anterior guardados", True)
        if usar:
            gal_lag = float(d.get("GALONES_ULTIMO_MES", 0) or 0)
            km_lag = float(d.get("KM_ULTIMO_MES", km_mes) or km_mes)
        else:
            a, b = st.columns(2)
            gal_lag = a.number_input("Galones del mes anterior", 0.0, value=30.0)
            km_lag = b.number_input("Kilómetros del mes anterior", 0.0, value=900.0)
        ejecutar = st.form_submit_button("Predecir", type="primary", use_container_width=True)
    if ejecutar:
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", periodo.strip()):
            st.error("Use el formato AAAA-MM.")
        else:
            fila = crear_fila(carro, clase, int(periodo[-2:]), km_mes, gal_lag, km_lag, clases_interaccion)
            esperado = max(0.0, float(modelo.predict(fila)[0]))
            a, b, c = st.columns(3)
            a.metric("Galones esperados", f"{esperado:,.1f}")
            b.metric("Costo esperado", dinero(esperado * precio_galon_base))
            c.metric("CO₂ estimado", f"{esperado * co2_galon:,.0f} kg")
            if facturados > 0:
                dif = facturados - esperado
                pct = dif / esperado if esperado else 0
                if dif > UMBRAL_REVISION_GAL and pct > UMBRAL_REVISION_PCT:
                    st.error(f"⚠️ Revisar: supera lo esperado en {dif:,.1f} galones ({pct:.0%}).")
                else:
                    st.success(f"✅ Consumo dentro del rango. Diferencia: {dif:,.1f} galones ({pct:.0%}).")

with tab2:
    st.subheader("Comparador para reemplazo de flota")
    st.write("Compara el costo de compra y energía durante el horizonte seleccionado. Los precios son editables.")
    with st.form("reemplazo"):
        c1, c2, c3 = st.columns(3)
        clase_r = c1.selectbox("Tipo de vehículo", ["AUTOMOVIL", "CAMIONETA"], format_func=lambda x: x.title())
        version = c2.selectbox("Versión gasolina", ["Básico", "Intermedio", "Full equipo"], index=1)
        ciudad = c3.selectbox("Tarifa eléctrica", list(tarifas.keys()), format_func=str.title)
        a, b, c, d = st.columns(4)
        cantidad = a.number_input("Vehículos a reemplazar", 1, 100, 5)
        anios = b.slider("Horizonte (años)", 1, 10, 5)
        km_anuales = c.number_input("Km por vehículo/año", 1000, 100000, 12000, 1000)
        mejora_gas = d.slider("Eficiencia del gasolina nuevo", 0, 30, 15, 5, help="Reducción estimada frente al consumo actual.")
        e, f, g = st.columns(3)
        consumo_actual = e.number_input("Rendimiento actual (km/galón)", 5.0, 100.0, 30.0, 1.0)
        precio_galon = f.number_input("Precio galón COP", 1000.0, 30000.0, float(round(precio_galon_base)), 500.0)
        margen_ev = g.slider("Consumo EV sobre ficha", 0, 40, 15, 5, help="Margen por tráfico, pendientes y condiciones reales.")
        nombre_gas, precio_gas_def = CARROS_GASOLINA[clase_r][version]
        h, i = st.columns(2)
        precio_gas = h.number_input(f"Precio {nombre_gas}", 10_000_000, 500_000_000, precio_gas_def, 1_000_000)
        precio_co2 = i.number_input("Valor máximo por tonelada de CO₂ evitada", 0, 10_000_000, 500_000, 100_000)
        comparar = st.form_submit_button("Comparar alternativas", type="primary", use_container_width=True)

    if comparar:
        tarifa = float(tarifas[ciudad])
        rendimiento_nuevo = consumo_actual / (1 - mejora_gas/100) if mejora_gas < 100 else consumo_actual
        gal_anuales = km_anuales / rendimiento_nuevo
        costo_energia_gas = gal_anuales * precio_galon * anios
        co2_total_gas = gal_anuales * co2_galon * anios / 1000
        total_gas = precio_gas + costo_energia_gas
        filas = []
        for nombre, info in CARROS_ELECTRICOS.items():
            if clase_r not in info["clases"]: continue
            ficha = vehiculos_ev.get(info["version"], {})
            kwh_100 = float(ficha.get("kwh_100km", 15.0)) * (1 + margen_ev/100)
            kwh_total = km_anuales * anios * kwh_100 / 100
            energia_ev = kwh_total * tarifa
            co2_ev = kwh_total * co2_kwh / 1000
            total_ev = info["precio"] + energia_ev
            ahorro = total_gas - total_ev
            co2_evitado = max(0, co2_total_gas - co2_ev)
            sobrecosto = max(0, total_ev - total_gas)
            costo_ton = sobrecosto / co2_evitado if co2_evitado > 0 else np.inf
            recomendado = ahorro >= 0 or costo_ton <= precio_co2
            filas.append({"Alternativa": nombre, "Compra": info["precio"], "Energía": energia_ev,
                "Costo total": total_ev, "CO₂ (t)": co2_ev, "Ahorro vs gasolina": ahorro,
                "CO₂ evitado (t)": co2_evitado, "Recomendado": "Sí" if recomendado else "No"})
        filas.append({"Alternativa": nombre_gas, "Compra": precio_gas, "Energía": costo_energia_gas,
            "Costo total": total_gas, "CO₂ (t)": co2_total_gas, "Ahorro vs gasolina": 0,
            "CO₂ evitado (t)": 0, "Recomendado": "Referencia"})
        tabla = pd.DataFrame(filas).sort_values("Costo total").reset_index(drop=True)
        mejor = tabla.iloc[0]
        x, y, z = st.columns(3)
        x.metric("Alternativa de menor costo", mejor["Alternativa"])
        y.metric("Costo unitario a término", millones(mejor["Costo total"]))
        z.metric("Costo total de la decisión", millones(mejor["Costo total"] * cantidad))
        st.dataframe(tabla.style.format({"Compra": dinero, "Energía": dinero, "Costo total": dinero,
            "CO₂ (t)": "{:,.2f}", "Ahorro vs gasolina": dinero, "CO₂ evitado (t)": "{:,.2f}"}),
            use_container_width=True, hide_index=True)
        resumen = tabla.copy()
        for col in ["Compra", "Energía", "Costo total", "Ahorro vs gasolina"]:
            resumen[col] = resumen[col] * cantidad
        resumen["CO₂ (t)"] *= cantidad
        resumen["CO₂ evitado (t)"] *= cantidad
        st.download_button("Descargar comparación CSV", resumen.to_csv(index=False).encode("utf-8-sig"),
            "comparacion_reemplazo_flota.csv", "text/csv", use_container_width=True)
        st.caption("La recomendación es financiera y ambiental. Antes de comprar, valide autonomía, carga útil, red de carga, garantía y cotizaciones vigentes.")

with st.sidebar:
    st.header("Información")
    st.write(f"Modelo: **{meta.get('nombre_modelo', 'No disponible')}**")
    st.write(f"Archivo: **{archivo_modelo}**")
    st.caption("Los precios de compra son valores de referencia editables y no constituyen una cotización comercial.")
