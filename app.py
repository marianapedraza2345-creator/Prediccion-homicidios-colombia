import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Homicidios Colombia",
    page_icon="📊",
    layout="wide"
)

# =========================
# DISEÑO PRO
# =========================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #071527, #0f172a);
    color: white;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
}
h1, h2, h3, p, label {
    color: #f8fafc !important;
}
.card {
    background: rgba(15, 23, 42, 0.95);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    box-shadow: 0px 10px 30px rgba(0,0,0,0.35);
}
.metric-card {
    background: linear-gradient(135deg, #111827, #1e293b);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(99,102,241,0.45);
    box-shadow: 0px 10px 25px rgba(0,0,0,0.35);
}
.metric-title {
    font-size: 14px;
    color: #93c5fd;
}
.metric-value {
    font-size: 36px;
    font-weight: 800;
    color: white;
}
.small-text {
    color: #cbd5e1;
    font-size: 14px;
}
.result-box {
    background: linear-gradient(135deg, #064e3b, #022c22);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid #22c55e;
}
.result-number {
    font-size: 48px;
    font-weight: 900;
    color: #bbf7d0;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CARGAR MODELO Y DATOS
# =========================

artefacto = joblib.load("mejor_modelo_homicidios.joblib")
modelo = artefacto["modelo_extra_trees"]
metricas = artefacto["metricas_test_2021"]

df = pd.read_excel("HOMICIDIO_Poblacion_InnerJoin.xlsx")
df.columns = df.columns.str.strip()

# Detectar columnas automáticamente
col_anio = None
col_homicidios = None
col_departamento = None

for col in df.columns:
    col_mayus = col.upper()

    if col_anio is None and ("AÑO" in col_mayus or "ANIO" in col_mayus):
        col_anio = col

    if col_homicidios is None and (
        "HOMICIDIO" in col_mayus or "CANTIDAD" in col_mayus
    ):
        col_homicidios = col

    if col_departamento is None and "DEPARTAMENTO" in col_mayus:
        col_departamento = col

# =========================
# SIDEBAR
# =========================

st.sidebar.markdown("## 📊 HOMICIDIOS")
st.sidebar.markdown("### COLOMBIA")
st.sidebar.markdown("---")

seccion = st.sidebar.radio(
    "Menú",
    ["🏠 Inicio", "🔮 Predicción", "📊 Análisis", "📋 Datos", "💡 Insights", "📄 Acerca del proyecto"]
)

st.sidebar.markdown("---")
st.sidebar.info("Proyecto académico de analítica aplicada.")

# =========================
# INICIO
# =========================

if seccion == "🏠 Inicio":

    col_title, col_status = st.columns([4, 1])

    with col_title:
        st.markdown("# Predicción de Homicidios en Colombia")
        st.markdown("Modelo predictivo basado en variables de población, educación y registros históricos.")

    with col_status:
        st.success("Modelo cargado")

    st.markdown("### Resumen del modelo")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MAPE (%)</div>
            <div class="metric-value">{metricas["MAPE (%)"]:.2f}</div>
            <div class="small-text">Error porcentual absoluto medio</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MAE</div>
            <div class="metric-value">{metricas["MAE"]:.2f}</div>
            <div class="small-text">Error absoluto medio</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">RMSE</div>
            <div class="metric-value">{metricas["RMSE"]:.2f}</div>
            <div class="small-text">Raíz del error cuadrático medio</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">R²</div>
            <div class="metric-value">{metricas["R2"]:.2f}</div>
            <div class="small-text">Coeficiente de determinación</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Vista general de los datos")

    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown(f"""
        <div class="card">
            <h3>👥 Registros totales</h3>
            <h2>{len(df):,}</h2>
            <p class="small-text">Observaciones en la base</p>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        deptos = df[col_departamento].nunique() if col_departamento else "N/A"
        st.markdown(f"""
        <div class="card">
            <h3>🗺️ Departamentos</h3>
            <h2>{deptos}</h2>
            <p class="small-text">Departamentos analizados</p>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        if col_anio:
            periodo = f"{df[col_anio].min()} - {df[col_anio].max()}"
        else:
            periodo = "N/A"

        st.markdown(f"""
        <div class="card">
            <h3>📅 Años analizados</h3>
            <h2>{periodo}</h2>
            <p class="small-text">Periodo de tiempo</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_graf, col_pred = st.columns([2, 1])

    with col_graf:
        st.markdown("### Evolución anual de homicidios")

        if col_anio and col_homicidios:
            anual = df.groupby(col_anio)[col_homicidios].sum().reset_index()

            fig, ax = plt.subplots(figsize=(9, 4))
            fig.patch.set_facecolor("#020617")
            ax.set_facecolor("#020617")
            ax.plot(anual[col_anio], anual[col_homicidios], marker="o", linewidth=3)
            ax.set_title("Evolución anual de homicidios", color="white")
            ax.set_xlabel("Año", color="white")
            ax.set_ylabel("Cantidad de homicidios", color="white")
            ax.tick_params(colors="white")
            ax.grid(alpha=0.2)

            st.pyplot(fig)
        else:
            st.markdown("""
            <div class="card">
                <h3>📈 Visualización pendiente</h3>
                <p>La base cargada no contiene columnas compatibles para construir esta gráfica automáticamente.</p>
            </div>
            """, unsafe_allow_html=True)

    with col_pred:
        st.markdown("### Predicción rápida")

        if col_departamento:
            departamentos = sorted(df[col_departamento].dropna().astype(str).unique())
        else:
            departamentos = ["ANTIOQUIA"]

        departamento = st.selectbox("Departamento", departamentos)
        anio = st.selectbox("Año", list(range(2022, 2031)))

        if st.button("🚀 Generar predicción"):
            entrada = pd.DataFrame([{
                "departamento": departamento.upper(),
                "anio_centrado": anio - 2005,
                "log_poblacion": 10,
                "tasa_lag_1": 50,
                "tasa_lag_2": 45,
                "homicidios_lag_1": 1000,
                "tasa_media_3_anios": 48,
                "tasa_media_5_anios": 47,
                "tasa_std_3_anios": 5,
                "cambio_tasa_1_anio": 2,
                "educacion_total_por_100k": 200,
                "universitaria_por_100k": 80,
                "posgrado_por_100k": 20,
                "TECNICA PROFESIONAL": 100,
                "TECNOLOGICA": 80,
                "UNIVERSITARIA": 200,
                "ESPECIALIZACION": 50,
                "MAESTRIA": 30,
                "DOCTORADO": 10
            }])

            pred_ml = modelo.predict(entrada)[0]
            alpha = artefacto["alpha_lag"]
            pred = alpha * entrada["tasa_lag_1"].values[0] + (1 - alpha) * pred_ml
            pred = max(pred, 0)

            st.markdown(f"""
            <div class="result-box">
                <p>Predicción estimada</p>
                <div class="result-number">{pred:.2f}</div>
                <p>homicidios por cada 100.000 habitantes</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Insights clave del proyecto")

    i1, i2, i3 = st.columns(3)

    with i1:
        st.markdown("""
        <div class="card">
            <h3>📍 Enfoque territorial</h3>
            <p>El análisis permite identificar diferencias significativas entre departamentos y priorizar zonas con mayor riesgo.</p>
        </div>
        """, unsafe_allow_html=True)

    with i2:
        st.markdown("""
        <div class="card">
            <h3>🎓 Factores educativos</h3>
            <p>El proyecto analiza la relación entre nivel educativo y tasa de homicidios por departamento.</p>
        </div>
        """, unsafe_allow_html=True)

    with i3:
        st.markdown("""
        <div class="card">
            <h3>🎯 Predicción inteligente</h3>
            <p>El modelo permite estimar escenarios futuros para apoyar decisiones públicas basadas en datos.</p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# PREDICCIÓN
# =========================

elif seccion == "🔮 Predicción":
    st.markdown("# 🔮 Simulador de Predicción")

    if col_departamento:
        departamentos = sorted(df[col_departamento].dropna().astype(str).unique())
    else:
        departamentos = ["ANTIOQUIA"]

    col1, col2 = st.columns(2)

    with col1:
        departamento = st.selectbox("Departamento", departamentos)

    with col2:
        anio = st.number_input("Año de predicción", 2022, 2030, 2022)

    if st.button("🚀 Generar predicción"):
        entrada = pd.DataFrame([{
            "departamento": departamento.upper(),
            "anio_centrado": anio - 2005,
            "log_poblacion": 10,
            "tasa_lag_1": 50,
            "tasa_lag_2": 45,
            "homicidios_lag_1": 1000,
            "tasa_media_3_anios": 48,
            "tasa_media_5_anios": 47,
            "tasa_std_3_anios": 5,
            "cambio_tasa_1_anio": 2,
            "educacion_total_por_100k": 200,
            "universitaria_por_100k": 80,
            "posgrado_por_100k": 20,
            "TECNICA PROFESIONAL": 100,
            "TECNOLOGICA": 80,
            "UNIVERSITARIA": 200,
            "ESPECIALIZACION": 50,
            "MAESTRIA": 30,
            "DOCTORADO": 10
        }])

        pred_ml = modelo.predict(entrada)[0]
        alpha = artefacto["alpha_lag"]
        pred = alpha * entrada["tasa_lag_1"].values[0] + (1 - alpha) * pred_ml
        pred = max(pred, 0)

        st.success(f"Predicción para {departamento} en {anio}: {pred:.2f} homicidios por cada 100.000 habitantes")

# =========================
# ANÁLISIS
# =========================

elif seccion == "📊 Análisis":
    st.markdown("# 📊 Análisis del Proyecto")

    st.markdown("""
    <div class="card">
        <h3>Objetivo de negocio</h3>
        <p>Analizar la evolución de los homicidios para priorizar acciones de intervención en seguridad pública.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Hallazgos del análisis exploratorio</h3>
        <p>La tasa de homicidios presentó una tendencia descendente entre 2005 y 2016, seguida de un incremento abrupto en 2018.</p>
        <p>También se evidenció alta variabilidad territorial, lo que muestra que el riesgo relativo no se distribuye de forma homogénea.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Modelado</h3>
        <p>Se probaron diferentes modelos predictivos y el modelo final utiliza una mezcla entre Extra Trees y una variable rezagada.</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# DATOS
# =========================

elif seccion == "📋 Datos":
    st.markdown("# 📋 Datos del Proyecto")

    st.write("Vista previa de la base de datos:")
    st.dataframe(df.head(20))

    st.write("Columnas disponibles:")
    st.write(df.columns.tolist())

# =========================
# INSIGHTS
# =========================

elif seccion == "💡 Insights":
    st.markdown("# 💡 Insights finales")

    st.success("""
    La violencia homicida presenta variaciones interanuales y alta concentración territorial.
    """)

    st.info("""
    Los homicidios afectan principalmente a hombres y se concentran en zonas urbanas.
    """)

    st.warning("""
    La relación entre educación y homicidios no es lineal, por lo que se requieren modelos capaces de capturar patrones complejos.
    """)

# =========================
# ACERCA
# =========================

elif seccion == "📄 Acerca del proyecto":
    st.markdown("# 📄 Acerca del Proyecto")

    st.markdown("""
    <div class="card">
        <h3>Proyecto de Analítica Aplicada</h3>
        <p>Este proyecto analiza la evolución de los homicidios en Colombia y su relación con variables poblacionales y educativas.</p>
        <p>El objetivo es apoyar la toma de decisiones mediante visualizaciones y predicciones basadas en datos.</p>
        <p><b>Equipo:</b> Juan Diego Ávila, Mariana Pedraza y Valentina Ordoñez.</p>
    </div>
    """, unsafe_allow_html=True)