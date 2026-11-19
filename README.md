# ⚡ Calculadora de Consumo de Combustible y Comparación de Movilidad Eléctrica

Aplicación web interactiva desarrollada en **Streamlit** para estimar la línea base de consumo de combustible (galones, costos en COP y emisiones de $\text{CO}_2$ operacional) de una flota vehicular y compararla frente a alternativas de vehículos eléctricos comercializados en Colombia.

---

## 📋 Descripción General

El sistema carga un modelo de regresión **Ridge con interacción ($\text{km} \times \text{clase}$)** previamente entrenado y serializado (`GrupoXX_Modelo_Final.pkl`), ejecutando inferencia en tiempo real **sin reentrenamiento**.

### Funcionalidades principales
* **🎛️ Simulador interactivo por vehículo:** Permite evaluar vehículos existentes en el histórico o vehículos nuevos (`VEH_NUEVO_2026`), ajustando clase, mes del año, kilómetros planificados, alternativa eléctrica, ciudad (tarifa de energía en COP/kWh) y factor de consumo real (tráfico, climatización y altitud).
* **🚚 Proyección de flota activa:** Proyecta el consumo esperado, gasto energético y huella de $\text{CO}_2$ del conjunto de vehículos activos en el último periodo registrado, permitiendo variar el factor de distancia.
* **📋 Casos de demostración:** Ejecuta escenarios predefinidos de referencia con visualización comparativa interactiva mediante **Plotly**.
* **📄 Ficha técnica y límites de uso:** Muestra las métricas de validación del modelo ($\text{MAE}$, $R^2$, $\text{MAPE}$, error total de flota) y las consideraciones metodológicas.

---

## 📁 Estructura del Repositorio

```text
├── app.py                      # Aplicación principal de Streamlit
├── requirements.txt            # Dependencias necesarias para el despliegue
├── GrupoXX_Modelo_Final.pkl    # Paquete serializado con el pipeline entrenado y metadatos
└── README.md                   # Documentación del proyecto
```

> **Nota:** Si el archivo del modelo tiene un nombre distinto (por ejemplo, `Grupo05_Modelo_Final.pkl`), puedes subirlo directamente en el repositorio con ese nombre y ajustarlo desde la barra lateral de la aplicación o en la constante `ARCHIVO_MODELO_DEFECTO` dentro de `app.py`.

---

## 🚀 Despliegue Online en Streamlit Community Cloud

1. Crea un repositorio público o privado en **GitHub** y sube los archivos:
   * `app.py`
   * `requirements.txt`
   * `GrupoXX_Modelo_Final.pkl`
   * `README.md`
2. Ingresa a [Streamlit Community Cloud](https://share.streamlit.io/) e inicia sesión con tu cuenta de GitHub.
3. Haz clic en **New app**, selecciona tu repositorio, la rama (`main`) y establece `app.py` como **Main file path**.
4. Haz clic en **Deploy**.

---

## 💻 Ejecución Local

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/tu-repositorio.git
   cd tu-repositorio
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecuta la aplicación:
   ```bash
   streamlit run app.py
   ```

---

## ⚠️ Alcance y Consideraciones Metodológicas

* **Línea base:** Los *galones esperados* representan una estimación estadística condicionada a la distancia recorrida y la clase del vehículo, no un límite operativo estricto.
* **Escenario eléctrico:** Utiliza consumos homologados o aproximados ($\text{kWh}/100\text{ km}$) y tarifas eléctricas de referencia por ciudad. El slider de *consumo real* permite incorporar hasta un $+30\%$ adicional por condiciones de tráfico, altitud y uso de aire acondicionado.
* **Límites del análisis:** No contempla costos de adquisición (CAPEX), infraestructura de carga, mantenimiento ni ciclo de vida de baterías. Las emisiones corresponden únicamente a $\text{CO}_2$ operacional.
