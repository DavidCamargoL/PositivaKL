from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

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


def generar_pdf_reposicion(tabla, cantidad, clase, anios, km_anuales, ciudad, nombre_gas,
                            mejor_ev, ahorro_flota, co2_ev_flota, reduccion_pct, decision,
                            precio_galon, tarifa):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.6*cm, leftMargin=1.6*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm,
                            title="Informe de reposición de flota")
    estilos = getSampleStyleSheet()
    morado = colors.HexColor("#4C1158")
    rosado = colors.HexColor("#D00075")
    gris = colors.HexColor("#F4F2F5")
    estilos.add(ParagraphStyle(name="TituloPositiva", parent=estilos["Title"], fontName="Helvetica-Bold",
                               fontSize=20, leading=24, textColor=morado, alignment=TA_CENTER, spaceAfter=10))
    estilos.add(ParagraphStyle(name="SubtituloPositiva", parent=estilos["Heading2"], fontName="Helvetica-Bold",
                               fontSize=13, leading=16, textColor=morado, spaceBefore=8, spaceAfter=7))
    estilos.add(ParagraphStyle(name="Resumen", parent=estilos["BodyText"], fontSize=10.5, leading=15,
                               backColor=gris, borderColor=morado, borderWidth=1, borderPadding=10,
                               spaceBefore=5, spaceAfter=12))
    historia = []
    historia.append(Paragraph("Plan de reposición de flota", estilos["TituloPositiva"]))
    historia.append(Paragraph("Comparación económica y ambiental entre vehículos de gasolina y eléctricos", estilos["BodyText"]))
    historia.append(Spacer(1, 10))

    efecto = "un ahorro" if ahorro_flota >= 0 else "un sobrecosto"
    resumen = (f"Para reemplazar <b>{cantidad} {clase.lower()}(s)</b> durante <b>{anios} años</b>, "
               f"la alternativa eléctrica de menor costo es <b>{mejor_ev['Alternativa']}</b>. "
               f"Frente a comprar <b>{nombre_gas}</b>, representa {efecto} de "
               f"<b>{dinero(abs(ahorro_flota))}</b> para la flota y evita aproximadamente "
               f"<b>{co2_ev_flota:,.1f} toneladas de CO2</b>, una reducción estimada de "
               f"<b>{reduccion_pct:,.0f}%</b>. La recomendación resultante es <b>{decision}</b>.")
    historia.append(Paragraph(resumen, estilos["Resumen"]))

    historia.append(Paragraph("Indicadores principales", estilos["SubtituloPositiva"]))
    indicadores = [
        ["Vehículos", "Horizonte", "Uso anual", "Alternativa recomendada"],
        [str(cantidad), f"{anios} años", f"{km_anuales:,.0f} km", str(decision)],
        ["Ahorro / sobrecosto", "CO2 evitado", "Reducción de CO2", "Ciudad tarifaria"],
        [dinero(ahorro_flota), f"{co2_ev_flota:,.1f} t", f"{reduccion_pct:,.0f}%", str(ciudad).title()],
    ]
    t = Table(indicadores, colWidths=[4.1*cm]*4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), morado), ("BACKGROUND", (0,2), (-1,2), rosado),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("TEXTCOLOR", (0,2), (-1,2), colors.white),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,2), (-1,2), "Helvetica-Bold"), ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("GRID", (0,0), (-1,-1), .5, colors.HexColor("#D5CBD7")),
        ("BACKGROUND", (0,1), (-1,1), colors.white), ("BACKGROUND", (0,3), (-1,3), gris),
        ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]))
    historia.append(t)
    historia.append(Spacer(1, 12))

    historia.append(Paragraph("Comparación de costos y emisiones", estilos["SubtituloPositiva"]))
    encabezados = ["Alternativa", "Compra", "Energía", "Costo total", "CO2 (t)", "CO2 evitado"]
    datos = [encabezados]
    for _, r in tabla.iterrows():
        datos.append([str(r["Alternativa"]), dinero(r["Compra"]*cantidad), dinero(r["Energía"]*cantidad),
                      dinero(r["Costo total"]*cantidad), f"{r['CO₂ (t)']*cantidad:,.1f}",
                      f"{r['CO₂ evitado (t)']*cantidad:,.1f}"])
    tc = Table(datos, repeatRows=1, colWidths=[4.0*cm, 2.8*cm, 2.8*cm, 3.0*cm, 1.8*cm, 2.1*cm])
    tc.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), morado), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8), ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#D5CBD7")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, gris]),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    historia.append(tc)
    historia.append(Spacer(1, 12))

    historia.append(Paragraph("Supuestos utilizados", estilos["SubtituloPositiva"]))
    supuestos = (f"El análisis usa {km_anuales:,.0f} km anuales por vehículo, un horizonte de {anios} años, "
                 f"un precio de gasolina de {dinero(precio_galon)} por galón y una tarifa eléctrica de "
                 f"{dinero(tarifa)} por kWh para {str(ciudad).title()}. Los costos representan compra y energía; "
                 "no incluyen cargadores, mantenimiento, seguros, impuestos, financiación ni beneficios tributarios.")
    historia.append(Paragraph(supuestos, estilos["BodyText"]))
    historia.append(Paragraph("Verificaciones antes de comprar", estilos["SubtituloPositiva"]))
    verificaciones = [
        "Confirmar autonomía real para los recorridos diarios.",
        "Validar puntos de carga e inversión de instalación en cada sede.",
        "Revisar capacidad de carga útil, especialmente para camionetas.",
        "Solicitar cotizaciones vigentes y comparar garantía, mantenimiento, seguros e impuestos.",
        "Realizar un piloto con un vehículo eléctrico antes de renovar toda la flota.",
    ]
    for i, texto in enumerate(verificaciones, 1):
        historia.append(Paragraph(f"<b>{i}.</b> {texto}", estilos["BodyText"]))
        historia.append(Spacer(1, 3))
    historia.append(Spacer(1, 8))
    historia.append(Paragraph("Informe generado automáticamente por la aplicación de gestión de flota.",
                               ParagraphStyle(name="Pie", parent=estilos["BodyText"], fontSize=8,
                                              textColor=colors.grey, alignment=TA_CENTER)))
    doc.build(historia)
    buffer.seek(0)
    return buffer.getvalue()

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
st.markdown("""
<style>
div[data-testid="stMetric"] {background:#f7f2f8; border:1px solid #d8bdd8; border-radius:14px; padding:14px;}
.resultado-principal {background:linear-gradient(135deg,#4c1158,#7b1b6f); color:white; padding:20px; border-radius:16px; margin:10px 0 18px 0;}
.resultado-principal h2 {margin:0 0 6px 0; color:white;}
.informe {background:#f8f9fb; border-left:6px solid #4c1158; padding:18px 20px; border-radius:10px; margin:14px 0;}
</style>
""", unsafe_allow_html=True)
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
            st.markdown(f"""<div class="resultado-principal"><h2>Resultado de la predicción</h2>
            <div>Para <b>{carro}</b>, el consumo esperado en <b>{periodo}</b> es de <b>{esperado:,.1f} galones</b>.</div></div>""", unsafe_allow_html=True)
            a, b, c = st.columns(3)
            a.metric("⛽ Galones esperados", f"{esperado:,.1f}")
            b.metric("💰 Costo esperado", dinero(esperado * precio_galon_base))
            c.metric("🌱 CO₂ estimado", f"{esperado * co2_galon:,.0f} kg")
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
        st.markdown("### Comparación visual de costos")
        grafico = tabla[["Alternativa", "Compra", "Energía"]].copy()
        grafico["Compra"] = grafico["Compra"] * cantidad / 1_000_000
        grafico["Energía"] = grafico["Energía"] * cantidad / 1_000_000
        grafico = grafico.set_index("Alternativa")
        st.bar_chart(grafico, horizontal=True, color=["#5b1765", "#d00075"],
            x_label="Costo para la flota (millones de COP)", y_label="Alternativa", stack=True)
        st.caption("La barra muestra el costo acumulado de compra y energía para todos los vehículos seleccionados.")

        st.markdown("### Resumen ejecutivo")
        gas_row = tabla[tabla["Alternativa"] == nombre_gas].iloc[0]
        mejor_ev_df = tabla[tabla["Alternativa"].isin(CARROS_ELECTRICOS.keys())].sort_values("Costo total")
        mejor_ev = mejor_ev_df.iloc[0]
        ahorro_flota = (gas_row["Costo total"] - mejor_ev["Costo total"]) * cantidad
        co2_ev_flota = mejor_ev["CO₂ evitado (t)"] * cantidad
        reduccion_pct = (mejor_ev["CO₂ evitado (t)"] / gas_row["CO₂ (t)"] * 100) if gas_row["CO₂ (t)"] else 0
        decision = mejor_ev["Alternativa"] if mejor_ev["Recomendado"] == "Sí" else nombre_gas
        efecto = "ahorraría" if ahorro_flota >= 0 else "costaría"
        valor_efecto = abs(ahorro_flota)
        texto_informe = f"""
        <div class="informe"><b>Informe corto de reposición</b><br><br>
        Para reemplazar <b>{cantidad} {clase_r.lower()}(s)</b>, con un uso de <b>{km_anuales:,} km por año</b>
        y un horizonte de <b>{anios} años</b>, la alternativa eléctrica de menor costo es
        <b>{mejor_ev['Alternativa']}</b>. Frente a comprar <b>{nombre_gas}</b>, esta opción
        <b>{efecto} {dinero(valor_efecto)}</b> para toda la flota y evitaría aproximadamente
        <b>{co2_ev_flota:,.1f} toneladas de CO₂</b>, equivalentes a una reducción de
        <b>{reduccion_pct:,.0f}%</b> durante el periodo.<br><br>
        <b>Recomendación:</b> {decision}. Antes de tomar la decisión final, se deben validar autonomía diaria,
        disponibilidad de cargadores, carga útil, mantenimiento, seguros, impuestos y cotizaciones comerciales vigentes.
        </div>"""
        st.markdown(texto_informe, unsafe_allow_html=True)

        st.markdown("### Detalle de alternativas")
        st.dataframe(tabla.style.format({"Compra": dinero, "Energía": dinero, "Costo total": dinero,
            "CO₂ (t)": "{:,.2f}", "Ahorro vs gasolina": dinero, "CO₂ evitado (t)": "{:,.2f}"}),
            use_container_width=True, hide_index=True)
        resumen = tabla.copy()
        for col in ["Compra", "Energía", "Costo total", "Ahorro vs gasolina"]:
            resumen[col] = resumen[col] * cantidad
        resumen["CO₂ (t)"] *= cantidad
        resumen["CO₂ evitado (t)"] *= cantidad
        pdf_informe = generar_pdf_reposicion(
            tabla, cantidad, clase_r, anios, km_anuales, ciudad, nombre_gas, mejor_ev,
            ahorro_flota, co2_ev_flota, reduccion_pct, decision, precio_galon, tarifa
        )
        dc1, dc2 = st.columns(2)
        dc1.download_button("📄 Descargar informe PDF", pdf_informe,
            "informe_reposicion_flota.pdf", "application/pdf", use_container_width=True)
        dc2.download_button("📊 Descargar comparación CSV", resumen.to_csv(index=False).encode("utf-8-sig"),
            "comparacion_reemplazo_flota.csv", "text/csv", use_container_width=True)
        st.caption("La recomendación es financiera y ambiental. Antes de comprar, valide autonomía, carga útil, red de carga, garantía y cotizaciones vigentes.")

with st.sidebar:
    st.header("Información")
    st.write(f"Modelo: **{meta.get('nombre_modelo', 'No disponible')}**")
    st.write(f"Archivo: **{archivo_modelo}**")
    st.caption("Los precios de compra son valores de referencia editables y no constituyen una cotización comercial.")
