# -----------------------------------------------------------------------------
# 0. INSTALACIÓN DE LIBRERÍAS E IMPORTACIÓN DE MÓDULOS
# -----------------------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CORPORATIVOS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Movica | Simulador de Estrategias de Tesorería", layout="wide")

st.markdown("""
    <style>
    .kpi-container { display: flex; gap: 15px; margin-bottom: 25px; flex-wrap: wrap; }
    .kpi-card {
        flex: 1;
        min-width: 200px;
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #186663;
    }
    .kpi-card.alert { border-left: 5px solid #CF8651; }
    .kpi-card.risk { border-left: 5px solid #D32F2F; }
    .kpi-card.neutral { border-left: 5px solid #637397; }
    .kpi-card.success { border-left: 5px solid #2E7D32; background-color: #F1F8E9; }
    .kpi-title { color: #637397; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 5px; }
    .kpi-value { color: #1E293B; font-size: 26px; font-weight: bold; margin: 0; }
    .kpi-desc { color: #9CA3AF; font-size: 11px; margin-top: 5px; }
    .section-title { color: #186663; font-weight: bold; margin-top: 20px; margin-bottom: 15px; font-size: 18px; border-bottom: 1px solid #E2E8F0; padding-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. AUTENTICACIÓN Y SEGURIDAD
# -----------------------------------------------------------------------------
def check_password():
    def password_entered():
        if st.session_state["pwd_input"] == "Movica2026":
            st.session_state["password_correct"] = True
            del st.session_state["pwd_input"] 
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<h3 style='color: #186663;'>Acceso Restringido - Tesorería Servitravel</h3>", unsafe_allow_html=True)
        st.text_input("Ingrese la clave corporativa", type="password", on_change=password_entered, key="pwd_input")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("<h3 style='color: #186663;'>Acceso Restringido - Tesorería Servitravel</h3>", unsafe_allow_html=True)
        st.text_input("Ingrese la clave corporativa", type="password", on_change=password_entered, key="pwd_input")
        st.error("Credenciales incorrectas.")
        return False
    return True

if not check_password():
    st.stop()

# -----------------------------------------------------------------------------
# 3. INGESTA DE DATOS HISTÓRICOS Y CONSTANTES (VECTORES ORIGINALES)
# -----------------------------------------------------------------------------
MESES = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago']
V = np.array([2339.69, 2512.91, 2640.69, 2690.62, 2703.83, 2427.66, 2355.18, 2678.09])
PROP = np.array([362.38, 408.54, 406.19, 389.13, 414.77, 332.36, 211.99, 461.12])
TERC = np.array([501.64, 707.37, 704.71, 692.31, 787.16, 922.69, 831.26, 877.46])
PROV = np.array([1147.78, 1101.58, 1203.97, 1272.75, 1190.99, 1000.37, 1007.47, 1045.19])
ADM = np.array([188.61, 190.77, 235.99, 220.61, 212.55, 209.99, 217.00, 201.19])
FAM = np.array([0.97, 0.94, 0.55, 0.55, 0.0, 0.61, 0.96, 0.0])
OI  = np.array([26.16, 29.15, 31.00, 25.31, 29.82, 64.08, 27.88, 25.81])
OG  = np.array([22.17, 18.08, 22.08, 23.86, 27.42, 23.58, 23.51, 24.66])
CF  = np.array([9.53, 9.77, 10.62, 9.33, 10.17, 10.70, 11.38, 13.69])

CD = PROP + TERC + PROV
RAI = V - CD - ADM - FAM + OI - OG - CF

BAL_AGO = dict(cash=161.73, cxc=3697.30, cxp=246.60)
TAX, DEP = 0.35, 9.0
T = 4  

V3 = V[-3:].mean()
CD3 = CD[-3:].mean()
DSO0 = BAL_AGO['cxc'] / (V3 / 30)
DPO0 = BAL_AGO['cxp'] / (CD3 / 30)
CD0 = CD[-1] / V[-1]
ADM0 = ADM[-1] / V[-1]
YTD_V, YTD_RAI = V.sum(), RAI.sum()
OTH = (OI - OG - CF - FAM).sum() / V.sum()
DAILY_OUT = 85.354
META_DIAS = 20
META_CAJA_MONTO = META_DIAS * DAILY_OUT  # Aprox $1,707 Millones COP

def lerp(a, b, k):
    return a + (b - a) * k / T

# -----------------------------------------------------------------------------
# 4. MOTOR DE SIMULACIÓN MONTE CARLO (LÓGICA ORIGINAL)
# -----------------------------------------------------------------------------
@st.cache_data
def run_monte_carlo(N, g_val, cd_val, adm_val, dso_val, dpo_val):
    rng = np.random.default_rng(2026)
    ks = np.arange(1, T + 1)

    g = rng.normal(g_val, 0.010, (N, 1))
    v = V3 * (1 + g)**ks * (1 + rng.normal(0, 0.05, (N, T)))

    cdp = lerp(CD0, cd_val, ks) + rng.normal(0, 0.010, (N, 1)) + rng.normal(0, 0.020, (N, T))
    ap = lerp(ADM0, adm_val, ks) + rng.normal(0, 0.006, (N, T))

    dsoT = rng.triangular(max(dso_val - 5.7, 5), dso_val, dso_val + 5.7, (N, 1))
    dpoT = rng.triangular(max(dpo_val - 1.5, 0.5), dpo_val, dpo_val + 3, (N, 1))

    c = v * cdp
    a = v * ap
    r = v - c - a + v * OTH
    ni = np.where(r > 0, r * (1 - TAX), r)

    hv = np.concatenate([np.tile(V[-2:], (N, 1)), v], 1)
    hc = np.concatenate([np.tile(CD[-2:], (N, 1)), c], 1)
    av = np.stack([hv[:, i:i+3].mean(1) for i in range(T)], 1)
    ac = np.stack([hc[:, i:i+3].mean(1) for i in range(T)], 1)

    dso = DSO0 + (dsoT - DSO0) * ks / T
    dpo = DPO0 + (dpoT - DPO0) * ks / T

    cxc = av / 30 * dso
    cxp = ac / 30 * dpo

    cp = np.concatenate([np.full((N, 1), BAL_AGO['cxc']), cxc[:, :-1]], 1)
    pp = np.concatenate([np.full((N, 1), BAL_AGO['cxp']), cxp[:, :-1]], 1)

    cash = BAL_AGO['cash'] + np.cumsum(ni + DEP - (cxc - cp) + (cxp - pp), 1)
    ann = (YTD_RAI + r.sum(1)) / (YTD_V + v.sum(1))
    
    return v, cash, ann, c, a

# -----------------------------------------------------------------------------
# 5. PANEL DE CONTROL Y DEFINICIÓN DE ESTRATEGIAS
# -----------------------------------------------------------------------------
st.title("Movica | Análisis de Impacto de Estrategias de Tesorería")

with st.sidebar:
    st.header("Parámetros Base (Sin Estrategias)")
    st.markdown("Defina el comportamiento inercial de la compañía antes de aplicar planes de mejora.")
    cd_base = st.slider("Costo Directo (% Vtas)", 80.0, 95.0, 88.5, 0.1) / 100
    adm_base = st.slider("Gastos Admin (% Vtas)", 5.0, 12.0, 8.24, 0.1) / 100
    dso_base = st.slider("Días Cartera (DSO)", 10.0, 60.0, 44.6, 1.0)
    dpo_base = st.slider("Días Proveedores (DPO)", 3.0, 30.0, 3.3, 1.0)
    g_base = st.slider("Crecimiento Mensual (%)", -5.0, 5.0, 0.2, 0.1) / 100
    simulaciones = st.selectbox("Iteraciones", [5000, 10000], index=0)

st.markdown("<div class='section-title'>Matriz de Ingeniería Financiera (Seleccione para evaluar impacto)</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    s1 = st.checkbox("1. Reingeniería de Pagos (Confirming): Incrementa DPO en 15 días mitigando salida de efectivo.")
    s2 = st.checkbox("2. Abono Mínimo Innegociable: Reduce DSO en 4 días mediante fraccionamiento de facturas altas.")
    s3 = st.checkbox("3. Factoraje Selectivo: Inyección agresiva. Reduce DSO 12 días, penaliza Admin en 0.5%.")
    s4 = st.checkbox("4. Optimización Estructural Híbrida: Vincula pago a flota con recaudo, reduce Costo Directo 1.0%.")
    s5 = st.checkbox("5. Distribución de Capacidad (Nov): Segmenta riesgo. Mejora crecimiento 1.0%, reduce DSO 2 días.")
with col2:
    s6 = st.checkbox("6. Scoring de Riesgo Crediticio: Filtra incobrables. Mejora DSO en 3 días de forma orgánica.")
    s7 = st.checkbox("7. Cobro Bipartito B2B (Anticipos): Fondea el 30% operativo. Reduce DSO drásticamente en 10 días.")
    s8 = st.checkbox("8. Descuentos Dinámicos (Pronto Pago): Reduce DSO 5 días, incrementa Gasto Financiero (Admin) 0.3%.")
    s9 = st.checkbox("9. Cash Pooling Restrictivo: Moviliza retenciones inerciales, asegura $50M fijos en reserva.")
    s10 = st.checkbox("10. Seguro de Crédito Comercial: Elimina volatilidad extrema. Costo 0.4% Admin.")

# Acumulador de impactos matemáticos sobre las variables base
dso_mod = dso_base
dpo_mod = dpo_base
cd_mod = cd_base
adm_mod = adm_base
g_mod = g_base
reserva_fija = 0

if s1: dpo_mod += 15.0
if s2: dso_mod -= 4.0
if s3: dso_mod -= 12.0; adm_mod += 0.005
if s4: cd_mod -= 0.010
if s5: g_mod += 0.010; dso_mod -= 2.0
if s6: dso_mod -= 3.0
if s7: dso_mod -= 10.0
if s8: dso_mod -= 5.0; adm_mod += 0.003
if s9: reserva_fija = 50.0
if s10: adm_mod += 0.004

dso_mod = max(1.0, dso_mod) # Límite matemático inferior para días de cartera

# -----------------------------------------------------------------------------
# 6. EJECUCIÓN DOBLE DEL MODELO (BASE VS ESTRATEGIA)
# -----------------------------------------------------------------------------
# Ejecución 1: Escenario Inercial (Sin estrategias activadas)
v_base_arr, cash_base_arr, ann_base_arr, _, _ = run_monte_carlo(simulaciones, g_base, cd_base, adm_base, dso_base, dpo_base)
caja_base_dic_p50 = np.percentile(cash_base_arr[:, -1], 50)

# Ejecución 2: Escenario Optimizado (Con estrategias aplicadas)
v_mod_arr, cash_mod_arr, ann_mod_arr, c_mod_arr, a_mod_arr = run_monte_carlo(simulaciones, g_mod, cd_mod, adm_mod, dso_mod, dpo_mod)
cash_mod_arr[:, -1] += reserva_fija # Aplicación de pooling a cierre
caja_mod_dic_p50 = np.percentile(cash_mod_arr[:, -1], 50)

# Cálculos de Impacto Monetario
diferencia_caja = caja_mod_dic_p50 - caja_base_dic_p50
prob_meta_mod = (cash_mod_arr[:, -1] >= META_CAJA_MONTO).mean() * 100
dias_caja_logrados = caja_mod_dic_p50 / DAILY_OUT
costos_totales_proyectados = (c_mod_arr.mean() + a_mod_arr.mean()) / v_mod_arr.mean() * 100

# -----------------------------------------------------------------------------
# 7. RENDERIZADO DE KPIs COMPARATIVOS
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>Resultados de Liquidez y Retorno de Estrategias</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card neutral">
        <div class="kpi-title">Requisito para 20 Días de Caja</div>
        <div class="kpi-value">${META_CAJA_MONTO:,.1f} M</div>
        <div class="kpi-desc">Monto innegociable a retener. ($85.3M diarios x 20)</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Caja Final Proyectada (Con Estrategias)</div>
        <div class="kpi-value">${caja_mod_dic_p50:,.1f} M</div>
        <div class="kpi-desc">Equivalente a {dias_caja_logrados:.1f} días de cobertura operativa.</div>
    </div>
    <div class="kpi-card success">
        <div class="kpi-title">Efectivo Liberado por Estrategias</div>
        <div class="kpi-value">${diferencia_caja:,.1f} M</div>
        <div class="kpi-desc">Capital de trabajo inyectado frente a no hacer nada.</div>
    </div>
    <div class="kpi-card {'alert' if prob_meta_mod < 50 else 'success'}">
        <div class="kpi-title">Probabilidad de Alcanzar 20 Días</div>
        <div class="kpi-value">{prob_meta_mod:.1f}%</div>
        <div class="kpi-desc">Certeza estadística bajo el mix de estrategias actual.</div>
    </div>
    <div class="kpi-card risk">
        <div class="kpi-title">Carga Variable Ajustada (800 + 1700)</div>
        <div class="kpi-value">{costos_totales_proyectados:.1f}%</div>
        <div class="kpi-desc">Proporción de ingresos absorbida tras optimizaciones.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. ANÁLISIS GRÁFICO COMPARATIVO
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>Trayectoria Comparativa de Efectivo (Base vs Optimizado)</div>", unsafe_allow_html=True)

ejes_x = ['Ago (Real)', 'Sep', 'Oct', 'Nov', 'Dic']

# Línea del escenario Base
p50_base = np.insert(np.percentile(cash_base_arr, 50, axis=0), 0, BAL_AGO['cash'])
# Percentiles del escenario Modificado (Estrategias)
percentiles_mod = np.percentile(cash_mod_arr, [5, 25, 50, 75, 95], axis=0)
p5_mod = np.insert(percentiles_mod[0], 0, BAL_AGO['cash'])
p25_mod = np.insert(percentiles_mod[1], 0, BAL_AGO['cash'])
p50_mod = np.insert(percentiles_mod[2], 0, BAL_AGO['cash'])
p75_mod = np.insert(percentiles_mod[3], 0, BAL_AGO['cash'])
p95_mod = np.insert(percentiles_mod[4], 0, BAL_AGO['cash'])

fig = go.Figure()
# Sombra de riesgo (Optimizado)
fig.add_trace(go.Scatter(x=ejes_x, y=p95_mod, mode='lines', line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=ejes_x, y=p5_mod, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(24, 102, 99, 0.1)', name='Rango 90% (Con Estrategias)'))
fig.add_trace(go.Scatter(x=ejes_x, y=p75_mod, mode='lines', line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=ejes_x, y=p25_mod, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(24, 102, 99, 0.25)', name='Rango 50% (Con Estrategias)'))

# Línea Inercial (Sin estrategias)
fig.add_trace(go.Scatter(x=ejes_x, y=p50_base, mode='lines+markers', line=dict(color='#CF8651', width=3, dash='dot'), name='Proyección Inercial (Sin hacer nada)'))
# Línea Optimizada (Con estrategias)
fig.add_trace(go.Scatter(x=ejes_x, y=p50_mod, mode='lines+markers', line=dict(color='#186663', width=3), name='Proyección Optimizada (Mediana)'))

# Línea de Meta (20 Días)
fig.add_hline(y=META_CAJA_MONTO, line_dash="dash", line_color="#D32F2F", annotation_text=f"Meta: {META_DIAS} Días ($1,707M)", annotation_position="top left")

fig.update_layout(
    yaxis_title="Disponibilidad de Caja (COP Millones)",
    hovermode="x unified", template="plotly_white", height=450, margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)
