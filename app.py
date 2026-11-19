from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Calculadora de Consumo y Movilidad Eléctrica",
    page_icon="⚡",
    layout="wide",
)

ARCHIVO_MODELO_DEFECTO = "GrupoXX_Modelo_Final.pkl"

MESES_NOMBRES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}


def buscar_ruta_modelo(nombre: str) -> Path | None:
    candidatos = [
        Path(nombre),
        Path("modelo") / nombre,
        Path("/content") / nombre,
    ]
    for ruta in candidatos:
        if ruta.exists():
            return ruta
    return None


@st.cache_resource(show_spinner="Cargando modelo entrenado...")
def cargar_paquete_desde_ruta(ruta_str: str):
    return joblib.load(ruta_str)


@st.cache_resource(show_spinner="Cargando modelo subido...")
def cargar_paquete_desde_bytes(archivo_bytes: bytes):
    import io
    return joblib.load(io.BytesIO(archivo_bytes))


def preparar_entorno(paquete: dict):
    modelo = paquete["modelo"]
    meta = paquete["metadata"]
    vehiculos = pd.DataFrame(meta["vehiculos"]).set_index("VEHICULO")
    externos = meta["parametros_externos"]
    fe_gas = externos["factores"]["gasolina_kg_co2_gal"] * meta["proporcion_gasolina_fosil"]
    fe_ele = externos["factores"]["electricidad_kg_co2_kwh"]
    return modelo, meta, vehiculos, externos, fe_gas, fe_ele


def construir_fila(vehiculo, clase, mes, km, galones_mes_anterior, km_mes_anterior, meta):
    fila = pd.DataFrame(
        [
            {
                "VEHICULO": vehiculo,
                "CLASE": clase,
                "KM_MES": float(km),
                "GALONES_LAG1": galones_mes_anterior,
                "KM_MES_LAG1": km_mes_anterior,
                "MES_SIN": np.sin(2 * np.pi * mes / 12),
                "MES_COS": np.cos(2 * np.pi * mes / 12),
            }
        ]
    )
    for c in meta["clases_interaccion"]:
        nombre = "KM_X_" + "".join(ch if ch.isalnum() else "_" for ch in str(c).upper())
        fila[nombre] = fila["KM_MES"] * (fila["CLASE"] == c)
    return fila


def validar(vehiculo, clase, mes, km, meta, vehiculos):
    if not 1 <= int(mes) <= 12:
        raise ValueError("El mes debe estar entre 1 y 12.")
    if km is None or float(km) <= 0:
        raise ValueError("Los kilómetros planificados deben ser mayores que cero.")

    alertas = []
    km_min, km_max = meta["rango_km_mes_entrenamiento"]
    if float(km) > float(km_max):
        alertas.append(
            f"La distancia supera el máximo observado ({km_max:,.0f} km): la estimación es una extrapolación."
        )
    if vehiculo not in vehiculos.index:
        alertas.append("Vehículo nuevo: se estima con su clase y la distancia.")
    if clase not in meta["clases_interaccion"]:
        alertas.append("Clase no vista en el entrenamiento: se usa la pendiente común de la distancia.")
    return alertas


def estimar(
    vehiculo,
    mes,
    km,
    modelo,
    meta,
    vehiculos,
    externos,
    fe_gas,
    fe_ele,
    clase=None,
    galones_mes_anterior=None,
    km_mes_anterior=None,
    alternativa="Tesla Model 3 Premium RWD",
    ciudad="BOGOTA",
    factor_consumo_ev=1.0,
):
    conocido = vehiculo in vehiculos.index
    clase_resuelta = clase or (vehiculos.loc[vehiculo, "CLASE"] if conocido else "AUTOMOVIL")

    if galones_mes_anterior is None and conocido:
        galones_mes_anterior = vehiculos.loc[vehiculo, "GALONES_ULTIMO_MES"]
    if km_mes_anterior is None and conocido:
        km_mes_anterior = vehiculos.loc[vehiculo, "KM_ULTIMO_MES"]

    alertas = validar(vehiculo, clase_resuelta, mes, km, meta, vehiculos)

    fila_x = construir_fila(
        vehiculo=vehiculo,
        clase=clase_resuelta,
        mes=mes,
        km=km,
        galones_mes_anterior=galones_mes_anterior,
        km_mes_anterior=km_mes_anterior,
        meta=meta,
    )
    galones = max(0.0, float(modelo.predict(fila_x)[0]))

    ev = externos["vehiculos_electricos"][alternativa]
    kwh = float(km) * ev["kwh_100km"] / 100.0 * float(factor_consumo_ev)
    tarifa = externos["tarifas_cop_kwh"][ciudad]

    costo_comb = galones * meta["precio_galon_referencia_cop"]
    co2_comb = galones * fe_gas
    costo_carga = kwh * tarifa
    co2_ele = kwh * fe_ele

    reduccion_co2_pct = (
        (1.0 - (co2_ele / co2_comb)) * 100.0 if (galones > 0 and co2_comb > 0) else np.nan
    )
    ahorro_cop = costo_comb - costo_carga

    return {
        "Vehículo": vehiculo,
        "Clase": clase_resuelta,
        "Mes": int(mes),
        "Km": float(km),
        "Galones esperados": galones,
        "Costo combustible COP": costo_comb,
        "CO2 combustión kg": co2_comb,
        "Alternativa": alternativa,
        "Ciudad": ciudad,
        "kWh": kwh,
        "Costo carga COP": costo_carga,
        "CO2 eléctrico kg": co2_ele,
        "Reducción CO2 %": reduccion_co2_pct,
        "Ahorro energía COP": ahorro_cop,
        "Alertas": " ".join(alertas) if alertas else "Sin alertas",
        "_lista_alertas": alertas,
        "_ev_meta": ev,
        "_tarifa": tarifa,
    }


def crear_graficos_comparativos(r: dict):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Costo de energía (COP)", "CO₂ operacional (kg)"),
        horizontal_spacing=0.14,
    )

    fig.add_trace(
        go.Bar(
            name="Costo",
            x=["Combustión", "Eléctrico"],
            y=[r["Costo combustible COP"], r["Costo carga COP"]],
            marker_color=["#C44E52", "#4C72B0"],
            text=[f"${r['Costo combustible COP']:,.0f}", f"${r['Costo carga COP']:,.0f}"],
            textposition="auto",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            name="CO₂",
            x=["Combustión", "Eléctrico"],
            y=[r["CO2 combustión kg"], r["CO2 eléctrico kg"]],
            marker_color=["#C44E52", "#55A868"],
            text=[f"{r['CO2 combustión kg']:,.1f} kg", f"{r['CO2 eléctrico kg']:,.1f} kg"],
            textposition="auto",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"{r['Vehículo']} · {r['Km']:,.0f} km · {r['Alternativa']} ({r['Ciudad']})",
        height=380,
        margin=dict(t=70, b=40, l=20, r=20),
        template="plotly_white",
    )
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    fig.update_yaxes(ticksuffix=" kg", row=1, col=2)
    return fig


def formatear_tabla_resultados(df: pd.DataFrame) -> pd.DataFrame:
    cols_visibles = [
        c for c in df.columns if not str(c).startswith("_")
    ]
    out = df[cols_visibles].copy()
    for col in ["Km", "Costo combustible COP", "Costo carga COP", "Ahorro energía COP"]:
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "-")
    for col in ["Galones esperados", "CO2 combustión kg", "kWh", "CO2 eléctrico kg"]:
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "-")
    if "Reducción CO2 %" in out.columns:
        out["Reducción CO2 %"] = out["Reducción CO2 %"].map(
            lambda v: f"{v:,.1f}%" if pd.notna(v) else "-"
        )
    return out


# --- Barra lateral: carga del modelo y parámetros de referencia ---
with st.sidebar:
    st.header("⚙️ Modelo y parámetros")
    nombre_archivo = st.text_input("Archivo del modelo (.pkl)", value=ARCHIVO_MODELO_DEFECTO)
    archivo_subido = st.file_uploader(
        "O sube tu archivo .pkl si no está en el repo",
        type=["pkl"],
    )

    paquete = None
    if archivo_subido is not None:
        paquete = cargar_paquete_desde_bytes(archivo_subido.getvalue())
    else:
        ruta_local = buscar_ruta_modelo(nombre_archivo)
        if ruta_local is not None:
            paquete = cargar_paquete_desde_ruta(str(ruta_local))

    if paquete is None:
        st.error(
            f"No se encontró `{nombre_archivo}` en el directorio. "
            "Súbelo aquí o colócalo en la misma carpeta que `app.py`."
        )
        st.stop()

    modelo, meta, vehiculos, externos, fe_gas, fe_ele = preparar_entorno(paquete)
    st.success("Modelo cargado sin reentrenar.")

    with st.expander("Parámetros de referencia", expanded=False):
        st.write(f"**Precio galón ref.:** ${meta['precio_galon_referencia_cop']:,.0f} COP")
        st.write(f"**FE gasolina fósil:** {fe_gas:.3f} kg CO₂/gal")
        st.write(f"**FE electricidad:** {fe_ele:.4f} kg CO₂/kWh")
        km_min, km_max = meta["rango_km_mes_entrenamiento"]
        st.write(f"**Rango km/mes entrenamiento:** {km_min:,.0f} – {km_max:,.0f} km")


# --- Encabezado ---
st.title("⚡ Calculadora de Consumo y Comparación Eléctrica")
st.caption(
    "Línea base de consumo (galones, COP y kg CO₂) y comparación frente a vehículos eléctricos "
    "comercializados en Colombia · Sin reentrenamiento en línea"
)

m = meta["metricas_prueba"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo", meta["nombre_modelo"])
c2.metric("MAE vehículo/mes", f"{m['vehiculo_mes']['MAE']:.2f} gal")
c3.metric("R² vehículo/mes", f"{m['vehiculo_mes']['R2']:.2f}")
c4.metric("Error total de flota", f"{m['error_flota_pct']:+.1f}%")

tab_sim, tab_flota, tab_casos, tab_ficha = st.tabs(
    [
        "🎛️ Simulador interactivo",
        "🚚 Proyección de flota activa",
        "📋 Casos de demostración",
        "📄 Ficha y límites",
    ]
)

# --- Pestaña 1: Simulador interactivo ---
with tab_sim:
    st.subheader("Simulador mensual por vehículo")

    opciones_veh = list(vehiculos.index) + ["VEH_NUEVO_2026"]
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        veh_sel = st.selectbox("Vehículo", options=opciones_veh, index=0)
        es_conocido = veh_sel in vehiculos.index
        clase_defecto = vehiculos.loc[veh_sel, "CLASE"] if es_conocido else "AUTOMOVIL"
        clases_disponibles = list(
            dict.fromkeys(list(meta["clases_interaccion"]) + [clase_defecto])
        )
        clase_sel = st.selectbox(
            "Clase del vehículo",
            options=clases_disponibles,
            index=clases_disponibles.index(clase_defecto),
        )

    with col_b:
        mes_sel = st.selectbox(
            "Mes del año",
            options=list(MESES_NOMBRES.keys()),
            format_func=lambda x: f"{x} · {MESES_NOMBRES[x]}",
            index=0,
        )
        km_sel = st.number_input(
            "Kilómetros planificados en el mes",
            min_value=1.0,
            max_value=25000.0,
            value=800.0,
            step=50.0,
        )

    with col_c:
        evs_disponibles = list(externos["vehiculos_electricos"].keys())
        idx_ev = evs_disponibles.index("Tesla Model 3 Premium RWD") if "Tesla Model 3 Premium RWD" in evs_disponibles else 0
        ev_sel = st.selectbox("Alternativa eléctrica", options=evs_disponibles, index=idx_ev)

        ciudades_disp = list(externos["tarifas_cop_kwh"].keys())
        ciudad_sel = st.selectbox("Ciudad (tarifa kWh)", options=ciudades_disp, index=0)

    factor_ev = st.slider(
        "Factor de consumo real eléctrico (tráfico, climatización, altitud)",
        min_value=1.00,
        max_value=1.30,
        value=1.00,
        step=0.05,
        help="1.00 equivale al consumo nominal/reportado; hasta 1.30 añade hasta +30%.",
    )

    with st.expander("Variables rezagadas opcionales (LAG1)", expanded=not es_conocido):
        lag_c1, lag_c2 = st.columns(2)
        gal_lag_def = float(vehiculos.loc[veh_sel, "GALONES_ULTIMO_MES"]) if es_conocido else 0.0
        km_lag_def = float(vehiculos.loc[veh_sel, "KM_ULTIMO_MES"]) if es_conocido else 0.0
        usar_lag_custom = st.checkbox("Especificar manualmente GALONES_LAG1 y KM_MES_LAG1", value=not es_conocido)
        gal_lag_val = lag_c1.number_input("Galones mes anterior", min_value=0.0, value=gal_lag_def, step=1.0, disabled=not usar_lag_custom)
        km_lag_val = lag_c2.number_input("Km mes anterior", min_value=0.0, value=km_lag_def, step=50.0, disabled=not usar_lag_custom)

    if st.button("Calcular estimación", type="primary", use_container_width=True):
        try:
            res = estimar(
                vehiculo=veh_sel,
                mes=mes_sel,
                km=km_sel,
                modelo=modelo,
                meta=meta,
                vehiculos=vehiculos,
                externos=externos,
                fe_gas=fe_gas,
                fe_ele=fe_ele,
                clase=clase_sel,
                galones_mes_anterior=gal_lag_val if usar_lag_custom else None,
                km_mes_anterior=km_lag_val if usar_lag_custom else None,
                alternativa=ev_sel,
                ciudad=ciudad_sel,
                factor_consumo_ev=factor_ev,
            )

            for alerta in res["_lista_alertas"]:
                st.warning(alerta)
            if not res["_lista_alertas"]:
                st.info("Sin alertas: parámetros dentro del rango observado.")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Galones esperados", f"{res['Galones esperados']:,.2f} gal")
            r2.metric(
                "Costo combustible",
                f"${res['Costo combustible COP']:,.0f} COP",
                delta=f"Ahorro EV: ${res['Ahorro energía COP']:,.0f} COP",
            )
            r3.metric("Costo carga eléctrica", f"${res['Costo carga COP']:,.0f} COP")
            reduccion_str = (
                f"{res['Reducción CO2 %']:,.1f}%"
                if pd.notna(res["Reducción CO2 %"])
                else "N/A"
            )
            r4.metric("Reducción CO₂ operacional", reduccion_str)

            st.plotly_chart(crear_graficos_comparativos(res), use_container_width=True)

            ev_info = res["_ev_meta"]
            st.caption(
                f"Alternativa: **{res['Alternativa']}** · {ev_info['kwh_100km']} kWh/100 km "
                f"(ciclo {ev_info['ciclo']}, {ev_info['calidad']}) · "
                f"Tarifa {res['Ciudad']}: ${res['_tarifa']:,.0f} COP/kWh."
            )

            df_detalle = pd.DataFrame(
                [{"Campo": k, "Valor": v} for k, v in res.items() if not str(k).startswith("_")]
            )
            st.dataframe(df_detalle, use_container_width=True, hide_index=True)

        except ValueError as err:
            st.error(f"Entrada no válida: {err}")


# --- Pestaña 2: Proyección de flota activa ---
with tab_flota:
    st.subheader("Proyección de vehículos activos en el último periodo")
    periodo_max = vehiculos["PERIODO"].max()
    activos = vehiculos[vehiculos["PERIODO"] == periodo_max]
    st.write(
        f"Vehículos activos en **{periodo_max}**: **{len(activos)}** de **{len(vehiculos)}** registrados."
    )

    f_col1, f_col2, f_col3 = st.columns(3)
    factor_dist = f_col1.slider(
        "FACTOR_DISTANCIA",
        min_value=0.50,
        max_value=2.00,
        value=1.00,
        step=0.05,
        help="1.00 mantiene los km del último mes; 1.10 equivale a +10% de distancia.",
    )
    mes_flota = f_col2.selectbox(
        "Mes proyectado",
        options=list(MESES_NOMBRES.keys()),
        format_func=lambda x: f"{x} · {MESES_NOMBRES[x]}",
        index=0,
        key="mes_flota",
    )
    ev_flota = f_col3.selectbox(
        "Alternativa eléctrica para la flota",
        options=list(externos["vehiculos_electricos"].keys()),
        index=0,
        key="ev_flota",
    )

    filas_flota = [
        estimar(
            vehiculo=v,
            mes=mes_flota,
            km=max(1.0, float(fila["KM_ULTIMO_MES"]) * factor_dist),
            modelo=modelo,
            meta=meta,
            vehiculos=vehiculos,
            externos=externos,
            fe_gas=fe_gas,
            fe_ele=fe_ele,
            alternativa=ev_flota,
        )
        for v, fila in activos.iterrows()
    ]
    df_flota = pd.DataFrame(filas_flota)
    totales = df_flota[
        [
            "Km",
            "Galones esperados",
            "Costo combustible COP",
            "CO2 combustión kg",
            "kWh",
            "Costo carga COP",
            "CO2 eléctrico kg",
            "Ahorro energía COP",
        ]
    ].sum()

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Distancia total flota", f"{totales['Km']:,.0f} km")
    t2.metric("Combustión esperada", f"{totales['Galones esperados']:,.1f} gal")
    t3.metric(
        "Costo energía (Comb. vs EV)",
        f"${totales['Costo combustible COP']:,.0f}",
        delta=f"vs ${totales['Costo carga COP']:,.0f} carga",
    )
    t4.metric(
        "CO₂ operacional (Comb. vs EV)",
        f"{totales['CO2 combustión kg']:,.0f} kg",
        delta=f"vs {totales['CO2 eléctrico kg']:,.0f} kg EV",
        delta_color="inverse",
    )

    st.dataframe(formatear_tabla_resultados(df_flota), use_container_width=True, hide_index=True)


# --- Pestaña 3: Casos de demostración ---
with tab_casos:
    st.subheader("Casos de demostración predefinidos")
    st.caption(
        "Incluye ejemplos de vehículos conocidos y un vehículo nuevo no presente en el histórico."
    )

    camioneta = (
        vehiculos[vehiculos["CLASE"] == "CAMIONETA"].index[0]
        if (vehiculos["CLASE"] == "CAMIONETA").any()
        else vehiculos.index[0]
    )
    automovil = (
        vehiculos[vehiculos["CLASE"] == "AUTOMOVIL"].index[0]
        if (vehiculos["CLASE"] == "AUTOMOVIL").any()
        else vehiculos.index[0]
    )
    ev_yuan = (
        "BYD Yuan UP 380 GS"
        if "BYD Yuan UP 380 GS" in externos["vehiculos_electricos"]
        else list(externos["vehiculos_electricos"].keys())[0]
    )
    ev_model_y = (
        "Tesla Model Y Premium RWD"
        if "Tesla Model Y Premium RWD" in externos["vehiculos_electricos"]
        else list(externos["vehiculos_electricos"].keys())[0]
    )
    ciudad_baq = "BARRANQUILLA" if "BARRANQUILLA" in externos["tarifas_cop_kwh"] else list(externos["tarifas_cop_kwh"].keys())[0]

    casos = [
        estimar(camioneta, mes=1, km=1000, modelo=modelo, meta=meta, vehiculos=vehiculos, externos=externos, fe_gas=fe_gas, fe_ele=fe_ele),
        estimar(automovil, mes=1, km=800, modelo=modelo, meta=meta, vehiculos=vehiculos, externos=externos, fe_gas=fe_gas, fe_ele=fe_ele, alternativa=ev_yuan),
        estimar(
            "VEH_NUEVO_2026",
            mes=1,
            km=1200,
            modelo=modelo,
            meta=meta,
            vehiculos=vehiculos,
            externos=externos,
            fe_gas=fe_gas,
            fe_ele=fe_ele,
            clase="AUTOMOVIL",
            galones_mes_anterior=None,
            km_mes_anterior=None,
            alternativa=ev_model_y,
            ciudad=ciudad_baq,
        ),
    ]

    df_casos = pd.DataFrame(casos)
    st.dataframe(formatear_tabla_resultados(df_casos), use_container_width=True, hide_index=True)

    caso_idx = st.radio(
        "Selecciona un caso para graficar:",
        options=[0, 1, 2],
        format_func=lambda i: f"Caso {i + 1}: {casos[i]['Vehículo']} ({casos[i]['Km']:,.0f} km)",
        horizontal=True,
    )
    st.plotly_chart(crear_graficos_comparativos(casos[caso_idx]), use_container_width=True)


# --- Pestaña 4: Ficha técnica y límites ---
with tab_ficha:
    st.subheader("Ficha técnica del modelo")
    ficha = pd.DataFrame(
        {
            "Campo": [
                "Modelo",
                "Hiperparámetros",
                "Datos de entrenamiento",
                "Año de prueba",
                "MAE vehículo mes",
                "R² vehículo mes",
                "MAPE vehículo año",
                "Error total de flota",
                "Clases conocidas",
                "Vehículos conocidos",
            ],
            "Valor": [
                meta["nombre_modelo"],
                str(meta["hiperparametros"]),
                " a ".join(meta["periodo_datos"]),
                str(m["anio"]),
                f"{m['vehiculo_mes']['MAE']:.2f} galones",
                f"{m['vehiculo_mes']['R2']:.2f}",
                f"{m['vehiculo_anio']['MAPE']:.1f} %",
                f"{m['error_flota_pct']:+.1f} %",
                ", ".join(meta["clases_interaccion"]),
                str(len(vehiculos)),
            ],
        }
    )
    st.dataframe(ficha, use_container_width=True, hide_index=True)

    st.subheader("Interpretación y límites de uso")
    st.markdown(
        """
* **Galones esperados** son una línea base condicionada a la distancia, no un límite de consumo. Una diferencia frente a lo real orienta la revisión logística; no demuestra desperdicio.
* El **escenario eléctrico** usa consumos homologados o aproximados y tarifas de muestra. El control de **consumo real** permite sumar hasta un 30% por tráfico, climatización y altitud.
* No incluye compra de vehículos, cargadores, mantenimiento ni ciclo de vida de baterías.
* Las emisiones son $\text{CO}_2$ operacional con los factores de la calculadora de huella; no incluyen $\text{CH}_4$ ni $\text{N}_2\text{O}$.
        """
    )
