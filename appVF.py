from pathlib import Path
from io import BytesIO
import re
import unicodedata
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

st.set_page_config(page_title="Gestión de flota", page_icon="🚗", layout="wide")
MODEL_FILES = [Path("Grupo08_Modelo_Final.pkl"), Path("Grupo08_Modelo_Final.txt")]

# Valores de referencia editables en la aplicación.
VEHICULOS_GASOLINA = {
    "AUTOMOVIL": {"nombre": "Nissan Versa ADVANCE CVT", "precio": 82_990_000},
    "CAMIONETA": {"nombre": "Nissan Frontier PRO-S 4x2", "precio": 144_990_000},
}
VEHICULOS_EV = {
    "AUTOMOVIL": [
        {"nombre": "BYD Yuan UP", "version": "BYD Yuan UP 380 GS", "precio": 99_990_000, "kwh_100": 13.9},
        {"nombre": "Tesla Model 3", "version": "Tesla Model 3 Premium RWD", "precio": 114_990_000, "kwh_100": 13.2},
    ],
    "CAMIONETA": [
        {"nombre": "BYD Yuan UP", "version": "BYD Yuan UP 380 GS", "precio": 99_990_000, "kwh_100": 13.9},
        {"nombre": "Tesla Model Y", "version": "Tesla Model Y Premium RWD", "precio": 119_990_000, "kwh_100": 15.7},
    ],
}
ALIASES = {
    "VEHICULO": ["VEHICULO", "PLACA", "ID_VEHICULO", "CODIGO_VEHICULO", "CARRO"],
    "SEDE": ["SEDE", "CIUDAD", "BASE", "UBICACION"],
    "CLASE": ["CLASE", "TIPO_VEHICULO", "TIPO", "CATEGORIA"],
    "KM": ["KM_MES", "KILOMETROS", "KM", "DISTANCIA_KM", "KM_RECORRIDOS"],
    "GALONES": ["GALONES", "GALONES_FACTURADOS", "CONSUMO_GALONES", "GALONES_MES", "COMBUSTIBLE_GALONES"],
    "COSTO": ["COSTO_COMBUSTIBLE", "COSTO", "VALOR_COMBUSTIBLE", "COSTO_COP"],
    "PERIODO": ["PERIODO", "FECHA", "MES", "ANO_MES"],
    "EDAD": ["EDAD", "ANTIGUEDAD", "ANOS_VEHICULO"],
    "EN_USO": ["EN_USO", "ACTIVO", "ESTADO"],
}

@st.cache_resource
def cargar_modelo():
    ruta = next((p for p in MODEL_FILES if p.exists()), None)
    if ruta is None:
        raise FileNotFoundError("No se encontró Grupo08_Modelo_Final.pkl.")
    paquete = joblib.load(ruta)
    return paquete["modelo"], paquete["metadata"], ruta.name

def normalizar_texto(x):
    texto = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().upper().strip()
    return re.sub(r"[^A-Z0-9]+", "_", texto).strip("_")

def dinero(v):
    signo = "-" if v < 0 else ""
    return signo + "$" + f"{abs(v):,.0f}".replace(",", ".")

def millones(v):
    return f"${v/1_000_000:,.1f} millones".replace(",", "X").replace(".", ",").replace("X", ".")

def leer_archivo(archivo):
    nombre = archivo.name.lower()
    if nombre.endswith(".csv"):
        contenido = archivo.getvalue()
        for enc in ["utf-8-sig", "latin-1"]:
            try:
                return pd.read_csv(BytesIO(contenido), encoding=enc, sep=None, engine="python")
            except Exception:
                pass
        raise ValueError("No fue posible leer el CSV.")
    return pd.read_excel(archivo, engine="xlrd" if nombre.endswith(".xls") else "openpyxl")

def sugerir_columna(columnas, campo):
    mapa = {normalizar_texto(c): c for c in columnas}
    for alias in ALIASES[campo]:
        if normalizar_texto(alias) in mapa:
            return mapa[normalizar_texto(alias)]
    return "No disponible"

def preparar_base(df, mapeo, catalogo_vehiculos):
    salida = pd.DataFrame()
    for campo, col in mapeo.items():
        if col != "No disponible":
            salida[campo] = df[col]
    obligatorias = ["VEHICULO", "KM", "GALONES"]
    faltantes = [c for c in obligatorias if c not in salida]
    if faltantes:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(faltantes))
    salida["VEHICULO"] = salida["VEHICULO"].astype(str).str.strip()
    # El archivo de nuevas observaciones solo necesita VEHICULO, PERIODO, KM_MES y GALONES_FACTURADOS.
    # CLASE y SEDE/SUCURSAL se recuperan de la metadata incluida en el modelo entrenado.
    catalogo = catalogo_vehiculos.copy()
    if not catalogo.empty:
        catalogo["VEHICULO"] = catalogo["VEHICULO"].astype(str).str.strip()
        catalogo = catalogo.rename(columns={"SUCURSAL": "SEDE_MODELO", "CLASE": "CLASE_MODELO"})
        disponibles = [c for c in ["VEHICULO", "CLASE_MODELO", "SEDE_MODELO"] if c in catalogo]
        salida = salida.merge(catalogo[disponibles].drop_duplicates("VEHICULO"), on="VEHICULO", how="left")
    if "CLASE" not in salida: salida["CLASE"] = salida.get("CLASE_MODELO", "AUTOMOVIL")
    elif "CLASE_MODELO" in salida: salida["CLASE"] = salida["CLASE"].fillna(salida["CLASE_MODELO"])
    if "SEDE" not in salida: salida["SEDE"] = salida.get("SEDE_MODELO", "BOGOTA")
    elif "SEDE_MODELO" in salida: salida["SEDE"] = salida["SEDE"].fillna(salida["SEDE_MODELO"])
    salida["CLASE"] = salida["CLASE"].map(normalizar_texto).replace({"AUTO":"AUTOMOVIL", "CARRO":"AUTOMOVIL", "SUV":"CAMIONETA", "PICKUP":"CAMIONETA"})
    # Las clases sin ficha técnica se tratan como automóvil solo para permitir la simulación;
    # la interfaz informa cuántos casos requieren validación operativa.
    salida["CLASE_ORIGINAL"] = salida["CLASE"]
    salida.loc[~salida["CLASE"].isin(["AUTOMOVIL", "CAMIONETA"]), "CLASE"] = "AUTOMOVIL"
    if "COSTO" not in salida: salida["COSTO"] = 0
    if "EDAD" not in salida: salida["EDAD"] = 0
    salida["SEDE"] = salida["SEDE"].astype(str).map(normalizar_texto)
    salida["KM"] = pd.to_numeric(salida["KM"], errors="coerce").fillna(0).clip(lower=0)
    salida["GALONES"] = pd.to_numeric(salida["GALONES"], errors="coerce").fillna(0).clip(lower=0)
    salida["COSTO"] = pd.to_numeric(salida["COSTO"], errors="coerce").fillna(0).clip(lower=0)
    salida["EDAD"] = pd.to_numeric(salida["EDAD"], errors="coerce").fillna(0).clip(lower=0)
    salida = salida[(salida["VEHICULO"] != "") & (salida["KM"] > 0)]
    if salida.empty:
        raise ValueError("No quedaron registros válidos después de la limpieza.")
    if "PERIODO" in salida:
        salida["PERIODO"] = salida["PERIODO"].astype(str)
    agrupada = salida.groupby("VEHICULO", as_index=False).agg(
        SEDE=("SEDE", "last"), CLASE=("CLASE", "last"), CLASE_ORIGINAL=("CLASE_ORIGINAL", "last"),
        KM_OBSERVADOS=("KM", "sum"), GALONES_OBSERVADOS=("GALONES", "sum"),
        COSTO_HISTORICO=("COSTO", "sum"), EDAD=("EDAD", "max"), REGISTROS=("VEHICULO", "size")
    )
    meses = salida.groupby("VEHICULO")["PERIODO"].nunique() if "PERIODO" in salida else salida.groupby("VEHICULO").size()
    agrupada["MESES_OBSERVADOS"] = agrupada["VEHICULO"].map(meses).clip(lower=1, upper=12)
    factor_anual = 12 / agrupada["MESES_OBSERVADOS"]
    agrupada["KM_ANUALES"] = agrupada["KM_OBSERVADOS"] * factor_anual
    agrupada["GALONES_ANUALES"] = agrupada["GALONES_OBSERVADOS"] * factor_anual
    agrupada["KM_GALON"] = np.where(agrupada["GALONES_OBSERVADOS"] > 0,
                                     agrupada["KM_OBSERVADOS"] / agrupada["GALONES_OBSERVADOS"], np.nan)
    mediana = agrupada["KM_GALON"].replace([np.inf, -np.inf], np.nan).median()
    agrupada["KM_GALON"] = agrupada["KM_GALON"].fillna(mediana if pd.notna(mediana) else 30).clip(5, 100)
    return agrupada

def analizar_flota(flota, anios, inicio, max_anual, mejora_gas, margen_ev, precio_galon,
                   tarifas, co2_galon, co2_kwh, costo_ton_max):
    filas = []
    for _, r in flota.iterrows():
        clase = r["CLASE"]
        gas = VEHICULOS_GASOLINA[clase]
        rendimiento_nuevo = r["KM_GALON"] / max(0.01, 1 - mejora_gas/100)
        gal_anual_nuevo = r["KM_ANUALES"] / rendimiento_nuevo
        costo_gas_energia = gal_anual_nuevo * precio_galon * anios
        co2_gas = gal_anual_nuevo * co2_galon * anios / 1000
        total_gas = gas["precio"] + costo_gas_energia
        tarifa = tarifas.get(r["SEDE"], np.mean(list(tarifas.values())))
        opciones = []
        for ev in VEHICULOS_EV[clase]:
            kwh100 = ev["kwh_100"] * (1 + margen_ev/100)
            kwh_total = r["KM_ANUALES"] * anios * kwh100 / 100
            energia = kwh_total * tarifa
            co2 = kwh_total * co2_kwh / 1000
            total = ev["precio"] + energia
            opciones.append((total, ev, energia, co2, kwh_total))
        total_ev, ev, energia_ev, co2_ev, kwh_total = min(opciones, key=lambda x: x[0])
        ahorro = total_gas - total_ev
        co2_evitado = max(0, co2_gas - co2_ev)
        sobrecosto = max(0, total_ev - total_gas)
        costo_ton = sobrecosto / co2_evitado if co2_evitado > 0 else np.inf
        viable_ev = ahorro >= 0 or costo_ton <= costo_ton_max
        score = (0.4 * co2_evitado + 0.3 * max(0, ahorro)/1_000_000 +
                 0.2 * r["KM_ANUALES"]/1000 + 0.1 * r["EDAD"])
        filas.append({
            **r.to_dict(), "GAS_MODELO": gas["nombre"], "GAS_COMPRA": gas["precio"],
            "GAS_ENERGIA": costo_gas_energia, "GAS_TOTAL": total_gas, "GAS_CO2": co2_gas,
            "GALONES_NUEVO": gal_anual_nuevo * anios, "EV_MODELO": ev["nombre"],
            "EV_COMPRA": ev["precio"], "EV_ENERGIA": energia_ev, "EV_TOTAL": total_ev,
            "EV_CO2": co2_ev, "KWH_TOTAL": kwh_total, "AHORRO_EV": ahorro,
            "CO2_EVITADO": co2_evitado, "COSTO_TON": costo_ton, "VIABLE_EV": viable_ev,
            "PUNTAJE": score, "TARIFA": tarifa,
        })
    detalle = pd.DataFrame(filas).sort_values("PUNTAJE", ascending=False).reset_index(drop=True)
    # Escalonar el plan. Los vehículos EV viables se priorizan; el resto pasa a gasolina.
    asignaciones = []
    for i, idx in enumerate(detalle.index):
        anio = inicio + min(i // max_anual, anios - 1)
        tecnologia = "Eléctrico" if detalle.loc[idx, "VIABLE_EV"] else "Gasolina"
        modelo = detalle.loc[idx, "EV_MODELO"] if tecnologia == "Eléctrico" else detalle.loc[idx, "GAS_MODELO"]
        asignaciones.append((anio, tecnologia, modelo))
    detalle[["ANIO_REPOSICION", "TECNOLOGIA", "MODELO_RECOMENDADO"]] = pd.DataFrame(asignaciones, index=detalle.index)

    todo_gas = {"ESCENARIO":"Todo gasolina", "COSTO":detalle.GAS_TOTAL.sum(), "CO2":detalle.GAS_CO2.sum(),
                "GALONES":detalle.GALONES_NUEVO.sum()}
    todo_ev = {"ESCENARIO":"Todo eléctrico", "COSTO":detalle.EV_TOTAL.sum(), "CO2":detalle.EV_CO2.sum(), "GALONES":0}
    costo_mixto = np.where(detalle.TECNOLOGIA.eq("Eléctrico"), detalle.EV_TOTAL, detalle.GAS_TOTAL).sum()
    co2_mixto = np.where(detalle.TECNOLOGIA.eq("Eléctrico"), detalle.EV_CO2, detalle.GAS_CO2).sum()
    gal_mixto = np.where(detalle.TECNOLOGIA.eq("Eléctrico"), 0, detalle.GALONES_NUEVO).sum()
    mixto = {"ESCENARIO":"Mixto recomendado", "COSTO":costo_mixto, "CO2":co2_mixto, "GALONES":gal_mixto}
    escenarios = pd.DataFrame([todo_gas, todo_ev, mixto])

    # Trayectoria anual según el año de reposición.
    anual = []
    for year in range(inicio, inicio + anios):
        costo = co2 = 0
        for _, r in detalle.iterrows():
            if year >= r.ANIO_REPOSICION and r.TECNOLOGIA == "Eléctrico":
                costo += r.EV_ENERGIA/anios
                co2 += r.EV_CO2/anios
            else:
                costo += r.GAS_ENERGIA/anios
                co2 += r.GAS_CO2/anios
        anual.append({"AÑO":year, "CO2_CON_PLAN":co2, "CO2_TODO_GAS":detalle.GAS_CO2.sum()/anios,
                      "COSTO_OPERATIVO_PLAN":costo})
    return detalle, escenarios, pd.DataFrame(anual)

def crear_pdf(escenarios, detalle, anual, parametros):
    b = BytesIO(); styles = getSampleStyleSheet(); morado = colors.HexColor("#4C1158"); verde = colors.HexColor("#2F8F5B")
    doc = SimpleDocTemplate(b, pagesize=A4, rightMargin=1.4*cm, leftMargin=1.4*cm, topMargin=1.3*cm, bottomMargin=1.3*cm)
    styles.add(ParagraphStyle(name="TituloX", parent=styles["Title"], textColor=morado, fontSize=20, leading=24))
    styles.add(ParagraphStyle(name="H2X", parent=styles["Heading2"], textColor=morado, fontSize=13, leading=16))
    story = [Paragraph("Plan de reposición de flota", styles["TituloX"]), Spacer(1,8)]
    gas = escenarios.iloc[0]; ev = escenarios.iloc[1]; mix = escenarios.iloc[2]
    ahorro = gas.COSTO - mix.COSTO; co2_ahorro = gas.CO2 - mix.CO2; pct = co2_ahorro/gas.CO2*100 if gas.CO2 else 0
    story += [Paragraph("Resumen ejecutivo", styles["H2X"]), Paragraph(
        f"El análisis comprende <b>{len(detalle)} vehículos</b> y un horizonte de <b>{parametros['anios']} años</b>. "
        f"El plan mixto cuesta <b>{dinero(mix.COSTO)}</b>, representa un ahorro de <b>{dinero(ahorro)}</b> frente al escenario todo gasolina "
        f"y evita <b>{co2_ahorro:,.1f} toneladas de CO2</b> ({pct:,.0f}% de reducción).", styles["BodyText"]), Spacer(1,10)]
    story += [Paragraph("1. Costo total y contaminación por escenario", styles["H2X"])]
    dt=[["Escenario","Costo total COP","CO2 (t)","Galones"]]+[[r.ESCENARIO,dinero(r.COSTO),f"{r.CO2:,.1f}",f"{r.GALONES:,.0f}"] for _,r in escenarios.iterrows()]
    t=Table(dt,colWidths=[4.3*cm,4.3*cm,3.2*cm,3.2*cm]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),morado),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.5,colors.grey),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F3F0F4")]),("ALIGN",(1,1),(-1,-1),"RIGHT"),("FONTSIZE",(0,0),(-1,-1),9),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)])); story += [t,Spacer(1,10)]
    story += [Paragraph("2. Contaminación que se deja de emitir", styles["H2X"]), Paragraph(
        f"El plan mixto evita <b>{co2_ahorro:,.1f} t de CO2</b>, reduce las emisiones en <b>{pct:,.0f}%</b> y deja de consumir "
        f"<b>{gas.GALONES-mix.GALONES:,.0f} galones</b> durante el horizonte evaluado.", styles["BodyText"]), Spacer(1,10)]
    story += [PageBreak(), Paragraph("3. Plan de reposición", styles["H2X"])]
    cols=["Año","Vehículo","Sede","Clase","Tecnología","Modelo","Costo","CO2 evitado"]
    rows=[cols]
    for _,r in detalle.iterrows():
        costo=r.EV_TOTAL if r.TECNOLOGIA=="Eléctrico" else r.GAS_TOTAL
        evit=r.CO2_EVITADO if r.TECNOLOGIA=="Eléctrico" else 0
        rows.append([str(int(r.ANIO_REPOSICION)),str(r.VEHICULO),str(r.SEDE),str(r.CLASE).title(),r.TECNOLOGIA,r.MODELO_RECOMENDADO,dinero(costo),f"{evit:,.1f} t"])
    tp=Table(rows,repeatRows=1,colWidths=[1.1*cm,2.4*cm,1.8*cm,1.8*cm,1.8*cm,3.1*cm,2.8*cm,1.6*cm]); tp.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),morado),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.35,colors.grey),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F3F0F4")]),("FONTSIZE",(0,0),(-1,-1),6.5),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)])); story += [tp,Spacer(1,8),Paragraph("Nota: los resultados son una simulación. Deben validarse autonomía, carga útil, infraestructura de carga, mantenimiento, seguros, impuestos y cotizaciones vigentes.",styles["BodyText"])]
    doc.build(story); b.seek(0); return b.getvalue()

def crear_fila_modelo(carro, clase, mes, km, gal_lag, km_lag, interacciones):
    fila=pd.DataFrame([{"VEHICULO":carro,"CLASE":clase,"KM_MES":km,"GALONES_LAG1":gal_lag,"KM_MES_LAG1":km_lag,"MES_SIN":np.sin(2*np.pi*mes/12),"MES_COS":np.cos(2*np.pi*mes/12)}])
    for c in interacciones:
        fila["KM_X_"+normalizar_texto(c)] = km*(clase==c)
    return fila

try:
    modelo, meta, archivo_modelo = cargar_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}"); st.stop()
externos=meta.get("parametros_externos",{}); factores=externos.get("factores",{})
tarifas_default={normalizar_texto(k):float(v) for k,v in externos.get("tarifas_cop_kwh",{"BOGOTA":1750,"BARRANQUILLA":1800}).items()}
precio_galon_default=float(meta.get("precio_galon_referencia_cop",16000)); proporcion=float(meta.get("proporcion_gasolina_fosil",1)); proporcion=proporcion/100 if proporcion>1 else proporcion
co2_galon=float(factores.get("gasolina_kg_co2_gal",8.813))*proporcion; co2_kwh=float(factores.get("electricidad_kg_co2_kwh",0.097))
catalogo_modelo=pd.DataFrame(meta.get("vehiculos",[]))
vehiculos_modelo=catalogo_modelo.set_index("VEHICULO")

st.markdown("""<style>
div[data-testid="stMetric"]{background:#f7f2f8;border:1px solid #d8bdd8;border-radius:14px;padding:14px}.hero{background:linear-gradient(135deg,#4c1158,#89156f);color:white;padding:20px;border-radius:16px;margin-bottom:16px}.hero h2{color:white;margin:0}.box{background:#eef8f1;border-left:6px solid #2f8f5b;padding:15px;border-radius:10px}</style>""",unsafe_allow_html=True)
st.title("🚗 Gestión y reposición de flota")
st.caption("Predicción individual y análisis completo de escenarios a partir de datos históricos.")
tab1,tab2=st.tabs(["⛽ Predicción individual","⚡ Análisis completo de flota"])

with tab1:
    with st.form("pred"):
        carro=st.selectbox("Vehículo",sorted(vehiculos_modelo.index.tolist())); d=vehiculos_modelo.loc[carro]; clase=str(d.get("CLASE","AUTOMOVIL"))
        c1,c2,c3=st.columns(3); periodo=c1.text_input("Periodo","2026-01"); km=c2.number_input("Kilómetros del mes",1.,10000.,900.,50.); fact=c3.number_input("Galones facturados",0.,500.,0.,1.)
        ejecutar=st.form_submit_button("Generar predicción",type="primary",use_container_width=True)
    if ejecutar:
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])",periodo): st.error("Use AAAA-MM.")
        else:
            fila=crear_fila_modelo(carro,clase,int(periodo[-2:]),km,float(d.get("GALONES_ULTIMO_MES",0) or 0),float(d.get("KM_ULTIMO_MES",km) or km),meta.get("clases_interaccion",[]))
            esperado=max(0,float(modelo.predict(fila)[0])); st.markdown(f'<div class="hero"><h2>{esperado:,.1f} galones esperados</h2><p>{carro} · {periodo}</p></div>',unsafe_allow_html=True)
            a,b,c=st.columns(3); a.metric("⛽ Consumo",f"{esperado:,.1f} gal"); b.metric("💰 Costo",dinero(esperado*precio_galon_default)); c.metric("🌱 CO2",f"{esperado*co2_galon:,.0f} kg")
            if fact>0:
                dif=fact-esperado; pct=dif/esperado if esperado else 0
                (st.error if dif>5 and pct>.25 else st.success)(f"Diferencia frente a facturación: {dif:,.1f} galones ({pct:.0%}).")

with tab2:
    st.subheader("Cargue de datos históricos")
    archivo=st.file_uploader("Cargue CSV o Excel",type=["csv","xlsx","xls"])
    st.caption("Columnas mínimas: vehículo, clase, kilómetros y galones. Puede haber varios meses por vehículo; la aplicación los consolida.")
    if archivo:
        try: bruto=leer_archivo(archivo)
        except Exception as e: st.error(str(e)); st.stop()
        st.write(f"Registros recibidos: **{len(bruto):,}** · Columnas: **{len(bruto.columns)}**")
        with st.expander("Mapear columnas",expanded=True):
            opciones=["No disponible"]+list(bruto.columns); mapeo={}; cols=st.columns(3)
            for i,campo in enumerate(ALIASES):
                sug=sugerir_columna(bruto.columns,campo); idx=opciones.index(sug) if sug in opciones else 0
                mapeo[campo]=cols[i%3].selectbox(campo,opciones,index=idx,key=f"map_{campo}")
        try: flota=preparar_base(bruto,mapeo,catalogo_modelo)
        except Exception as e: st.error(str(e)); st.stop()
        st.success(f"Base consolidada: {len(flota):,} vehículos válidos. Los datos observados se anualizaron según los meses disponibles.")
        sin_ficha = int((flota["CLASE_ORIGINAL"] == "SIN_FICHA_TECNICA").sum())
        if sin_ficha:
            st.warning(f"{sin_ficha} vehículos aparecen como 'SIN FICHA TÉCNICA' en la metadata. Se simulan como automóvil, pero deben validarse antes de aprobar el plan.")
        st.dataframe(flota[["VEHICULO","SEDE","CLASE_ORIGINAL","MESES_OBSERVADOS","KM_ANUALES","GALONES_ANUALES","KM_GALON"]].head(20), use_container_width=True, hide_index=True)
        with st.form("parametros"):
            st.markdown("#### Parámetros del análisis")
            c1,c2,c3,c4=st.columns(4)
            anios=c1.slider("Horizonte",1,10,5); inicio=c2.number_input("Año inicial",2024,2040,2026); max_anual=c3.number_input("Máximo reemplazos/año",1,500,max(1,int(np.ceil(len(flota)/anios)))); mejora=c4.slider("Mejora gasolina",0,30,15,5)
            d1,d2,d3,d4=st.columns(4)
            margen=d1.slider("Margen consumo EV",0,40,15,5); precio_g=d2.number_input("Precio galón COP",1000.,30000.,precio_galon_default,500.); costo_ton=d3.number_input("Máximo COP/t CO2",0.,10000000.,500000.,100000.); tarifa_otras=d4.number_input("Tarifa otras sedes COP/kWh",500.,5000.,float(np.mean(list(tarifas_default.values()))),50.)
            ejecutar_plan=st.form_submit_button("Generar escenarios y plan",type="primary",use_container_width=True)
        if ejecutar_plan:
            tarifas=tarifas_default.copy()
            for sede in flota.SEDE.unique(): tarifas.setdefault(sede,tarifa_otras)
            detalle,escenarios,anual=analizar_flota(flota,int(anios),int(inicio),int(max_anual),mejora,margen,precio_g,tarifas,co2_galon,co2_kwh,costo_ton)
            gas,ev,mix=escenarios.iloc[0],escenarios.iloc[1],escenarios.iloc[2]
            ahorro=gas.COSTO-mix.COSTO; co2_ahorro=gas.CO2-mix.CO2; reduccion=co2_ahorro/gas.CO2*100 if gas.CO2 else 0; gal_ahorro=gas.GALONES-mix.GALONES
            st.markdown("### 1. Costo total y CO2 por escenario")
            a,b,c=st.columns(3)
            for col,r,color in [(a,gas,"🟫"),(b,ev,"🟦"),(c,mix,"🟩")]:
                col.metric(f"{color} {r.ESCENARIO}",millones(r.COSTO),f"{r.CO2:,.1f} t CO2")
            graf_cost=escenarios.set_index("ESCENARIO")[["COSTO"]]/1_000_000; st.bar_chart(graf_cost,color="#6A1B6B",horizontal=True,x_label="Millones de COP")
            graf_co2=escenarios.set_index("ESCENARIO")[["CO2"]]; st.bar_chart(graf_co2,color="#2F8F5B",horizontal=True,x_label="Toneladas de CO2")
            st.markdown("### 2. Contaminación que se deja de emitir")
            x1,x2,x3,x4=st.columns(4); x1.metric("CO2 evitado",f"{co2_ahorro:,.1f} t"); x2.metric("Reducción",f"{reduccion:,.0f}%"); x3.metric("Galones no quemados",f"{gal_ahorro:,.0f}"); x4.metric("Ahorro del plan",dinero(ahorro))
            st.line_chart(anual.set_index("AÑO")[["CO2_TODO_GAS","CO2_CON_PLAN"]],color=["#B87333","#2F8F5B"],y_label="Toneladas de CO2 por año")
            top=detalle.nlargest(10,"CO2_EVITADO").set_index("VEHICULO")[["CO2_EVITADO"]]; st.bar_chart(top,color="#2F8F5B",horizontal=True,x_label="Toneladas de CO2 evitadas")
            st.markdown("### 3. Plan de reposición")
            vista=detalle[["ANIO_REPOSICION","VEHICULO","SEDE","CLASE","TECNOLOGIA","MODELO_RECOMENDADO","PUNTAJE","CO2_EVITADO","AHORRO_EV"]].copy()
            vista.columns=["Año","Vehículo","Sede","Clase","Tecnología","Modelo recomendado","Prioridad","CO2 evitado (t)","Ahorro EV"]
            st.dataframe(vista.style.format({"Prioridad":"{:,.1f}","CO2 evitado (t)":"{:,.1f}","Ahorro EV":dinero}),use_container_width=True,hide_index=True)
            st.markdown(f'<div class="box"><b>Resumen ejecutivo.</b> El plan mixto para {len(detalle)} vehículos cuesta <b>{dinero(mix.COSTO)}</b>. Frente al escenario todo gasolina, genera un ahorro de <b>{dinero(ahorro)}</b>, evita <b>{co2_ahorro:,.1f} toneladas de CO2</b> y reduce las emisiones en <b>{reduccion:,.0f}%</b>.</div>',unsafe_allow_html=True)
            pdf=crear_pdf(escenarios,detalle,anual,{"anios":anios})
            descarga=detalle.copy(); descarga.to_csv(index=False)
            q1,q2=st.columns(2); q1.download_button("📄 Descargar informe PDF",pdf,"plan_reposicion_flota.pdf","application/pdf",use_container_width=True); q2.download_button("📊 Descargar plan CSV",descarga.to_csv(index=False).encode("utf-8-sig"),"plan_reposicion_flota.csv","text/csv",use_container_width=True)
            st.warning("Los valores de compra y eficiencia son referencias editables. Valide cotizaciones, autonomía, carga útil, infraestructura, seguros, mantenimiento e impuestos antes de decidir.")

with st.sidebar:
    st.header("Modelo cargado"); st.write(archivo_modelo); st.caption("La pestaña 2 consolida múltiples registros por vehículo y construye tres escenarios completos.")
