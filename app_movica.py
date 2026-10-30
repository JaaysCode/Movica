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
st.set_page_config(page_title="Movica | Simulador y Estrategias", layout="wide")

st.markdown("""
    <style>
    /* Estilos generales y contenedores */
    .main-header { color: #186663; font-size: 28px; font-weight: bold; margin-bottom: 5px; border-bottom: 2px solid #186663; padding-bottom: 10px; }
    .sub-header { color: #637397; font-size: 16px; margin-bottom: 20px; }
    .section-title { color: #186663; font-size: 20px; font-weight: bold; margin-top: 40px; margin-bottom: 15px; border-bottom: 1px solid #E2E8F0; padding-bottom: 5px; }
    
    /* Cajas de texto instructivo */
    .instruction-box { background-color: #F8FAFC; border-left: 4px solid #186663; padding: 20px; margin-bottom: 25px; border-radius: 4px; color: #334155; font-size: 14px; line-height: 1.6; }
    .strategy-desc-box { background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 15px; margin-bottom: 10px; border-radius: 6px; border-left: 4px solid #CF8651; font-size: 13px; color: #475569; }
    
    /* Tarjetas KPI Flexibles para evitar recortes */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 25px; }
    .kpi-card { background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); border-left: 5px solid #186663; display: flex; flex-direction: column; justify-content: space-between; }
    .kpi-card.alert { border-left: 5px solid #CF8651; }
    .kpi-card.risk { border-left: 5px solid #D32F2F; }
    .kpi-card.neutral { border-left: 5px solid #637397; }
    .kpi-card.success { border-left: 5px solid #2E7D32; background-color: #F1F8E9; }
    .kpi-title { color: #637397; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; line-height: 1.2; }
    .kpi-value { color: #1E293B; font-size: 26px; font-weight: bold; margin: 0; }
    .kpi-desc { color: #9CA3AF; font-size: 11px; margin-top: 10px; line-height: 1.2; }
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
        st.markdown("<div class='main-header'>Acceso Restringido - Tesorería Servitravel</div>", unsafe_allow_html=True)
        st.text_input("Ingrese la clave corporativa", type="password", on_change=password_entered, key="pwd_input")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("<div class='main-header'>Acceso Restringido - Tesorería Servitravel</div>", unsafe_allow_html=True)
        st.text_input("Ingrese la clave corporativa", type="password", on_change=password_entered, key="pwd_input")
        st.error("Credenciales incorrectas.")
        return False
    return True

if not check_password():
    st.stop()

# -----------------------------------------------------------------------------
# 3. INGESTA DE DATOS HISTÓRICOS Y CONSTANTES
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
TAX, DEP, T = 0.35, 9.0, 4  

V3 = V[-3:].mean()
CD3 = CD[-3:].mean()
DSO0 = BAL_AGO['cxc'] / (V3 / 30)
DPO0 = BAL_AGO['cxp'] / (CD3 / 30)
CD0, ADM0 = CD[-1] / V[-1], ADM[-1] / V[-1]
YTD_V, YTD_RAI = V.sum(), RAI.sum()
OTH = (OI - OG - CF - FAM).sum() / V.sum()
DAILY_OUT = 85.354
META_DIAS = 20
META_CAJA_MONTO = META_DIAS * DAILY_OUT  

def lerp(a, b, k):
    return a + (b - a) * k / T

# -----------------------------------------------------------------------------
# 4. MOTOR DE SIMULACIÓN MONTE CARLO
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
# 5. CABECERA E INSTRUCTIVO GENERAL
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>Movica | Ecosistema de Modelación Financiera y Tesorería</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Plataforma analítica para Servitravel enfocada en la gestión de capital de trabajo y riesgo de liquidez.</div>", unsafe_allow_html=True)

st.markdown("""
<div class='instruction-box'>
    <b>Fundamento del Modelo Estocástico (Simulación de Monte Carlo)</b><br><br>
    El presente aplicativo tiene como objetivo evaluar la viabilidad de mantener una reserva de tesorería equivalente a <b>20 días de operación continua</b> (estimada en $1,707 millones de COP). Debido a la incertidumbre inherente en los ciclos de recaudo (cartera vencida) y las fluctuaciones en la demanda de la flota tercerizada, el modelo genera 10,000 escenarios probabilísticos posibles para cada mes restante del año.<br><br>
    <b>Metodología de lectura:</b> El sistema se divide en dos módulos. El Módulo 1 proyecta la situación financiera bajo diferentes escenarios de mercado predefinidos (Base, Optimista, Pesimista). El Módulo 2 permite a la junta directiva activar estrategias de ingeniería financiera para visualizar el impacto matemático directo sobre el escenario seleccionado en el Módulo 1.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. BLOQUE 1: SIMULADOR DE ESCENARIOS MACROECONÓMICOS
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>Módulo 1: Proyección de Tesorería por Escenario de Mercado</div>", unsafe_allow_html=True)

with st.expander("Panel de Control de Escenarios (Ajuste de Parámetros)", expanded=True):
    escenario = st.selectbox("Seleccione el Escenario Macroeconómico a Evaluar:", 
                             ['Personalizado', 'Base', 'Pesimista', 'Optimista'], index=0)
    
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 15px;'>", unsafe_allow_html=True)
    
    if escenario == 'Base':
        cd_min, cd_max, cd_val = 88.0, 89.0, 88.5
        adm_min, adm_max, adm_val = 8.0, 8.5, 8.24
        g_min, g_max, g_val_pct = 0.0, 0.5, 0.2
        dso_min, dso_max, dso_val = 21.0, 22.0, 21.5
        dpo_min, dpo_max, dpo_val = 14.0, 16.0, 15.0
    elif escenario == 'Pesimista':
        cd_min, cd_max, cd_val = 91.6, 93.0, 92.3
        adm_min, adm_max, adm_val = 9.0, 9.5, 9.21
        g_min, g_max, g_val_pct = -2.0, -1.0, -1.5
        dso_min, dso_max, dso_val = 25.0, 30.0, 27.5
        dpo_min, dpo_max, dpo_val = 10.0, 12.0, 11.0
    elif escenario == 'Optimista':
        cd_min, cd_max, cd_val = 84.0, 85.0, 84.5
        adm_min, adm_max, adm_val = 7.0, 7.6, 7.51
        g_min, g_max, g_val_pct = 1.5, 2.5, 2.0
        dso_min, dso_max, dso_val = 18.0, 19.0, 18.5
        dpo_min, dpo_max, dpo_val = 22.0, 25.0, 23.5
    else: 
        cd_min, cd_max, cd_val = 80.0, 95.0, 88.5
        adm_min, adm_max, adm_val = 5.0, 12.0, 8.24
        g_min, g_max, g_val_pct = -5.0, 5.0, 0.2
        dso_min, dso_max, dso_val = 10.0, 60.0, 44.6
        dpo_min, dpo_max, dpo_val = 3.0, 45.0, 3.3

    col_param1, col_param2, col_param3 = st.columns(3)
    with col_param1:
        cd_base = st.slider("Costo Directo (% Vtas)", min_value=cd_min, max_value=cd_max, value=cd_val, step=0.1) / 100
        adm_base = st.slider("Gastos Admin (% Vtas)", min_value=adm_min, max_value=adm_max, value=adm_val, step=0.1) / 100
    with col_param2:
        dso_base = st.slider("Días Cartera (DSO)", min_value=float(dso_min), max_value=float(dso_max), value=float(dso_val), step=0.5)
        dpo_base = st.slider("Días Proveedores (DPO)", min_value=float(dpo_min), max_value=float(dpo_max), value=float(dpo_val), step=0.5)
    with col_param3:
        g_base = st.slider("Crecimiento Mensual (%)", min_value=g_min, max_value=g_max, value=g_val_pct, step=0.1) / 100
        simulaciones = st.selectbox("Iteraciones (Precisión Monte Carlo)", [5000, 10000, 20000], index=1)

v_base_arr, cash_base_arr, ann_base_arr, c_base_arr, a_base_arr = run_monte_carlo(simulaciones, g_base, cd_base, adm_base, dso_base, dpo_base)
caja_base_dic_p50 = np.percentile(cash_base_arr[:, -1], 50)
prob_meta_base = (cash_base_arr[:, -1] >= META_CAJA_MONTO).mean() * 100
prob_deficit_base = (cash_base_arr.min(1) < 0).mean() * 100
costo_integrado_base = (c_base_arr.mean() + a_base_arr.mean()) / v_base_arr.mean() * 100

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card neutral">
        <div class="kpi-title">Proyección a Diciembre (P50)</div>
        <div class="kpi-value">${caja_base_dic_p50:,.1f} M</div>
        <div class="kpi-desc">Efectivo proyectado bajo el escenario <b>{escenario}</b>.</div>
    </div>
    <div class="kpi-card risk">
        <div class="kpi-title">Riesgo de Déficit de Caja (< $0)</div>
        <div class="kpi-value">{prob_deficit_base:.1f}%</div>
        <div class="kpi-desc">Probabilidad de insolvencia temporal operativa.</div>
    </div>
    <div class="kpi-card {'alert' if prob_meta_base < 50 else 'success'}">
        <div class="kpi-title">Probabilidad Cumplimiento (20 Días)</div>
        <div class="kpi-value">{prob_meta_base:.1f}%</div>
        <div class="kpi-desc">Certeza matemática de llegar a $1,707M.</div>
    </div>
    <div class="kpi-card risk">
        <div class="kpi-title">Carga Operativa Integrada</div>
        <div class="kpi-value">{costo_integrado_base:.1f}%</div>
        <div class="kpi-desc">Proporción de ingresos absorbida por Costos + Admin.</div>
    </div>
</div>
""", unsafe_allow_html=True)

percentiles_base = np.percentile(cash_base_arr, [5, 25, 50, 75, 95], axis=0)
ejes_x = ['Ago (Real)', 'Sep', 'Oct', 'Nov', 'Dic']
p5_base = np.insert(percentiles_base[0], 0, BAL_AGO['cash'])
p25_base = np.insert(percentiles_base[1], 0, BAL_AGO['cash'])
p50_base = np.insert(percentiles_base[2], 0, BAL_AGO['cash'])
p75_base = np.insert(percentiles_base[3], 0, BAL_AGO['cash'])
p95_base = np.insert(percentiles_base[4], 0, BAL_AGO['cash'])

fig_base = go.Figure()
fig_base.add_trace(go.Scatter(x=ejes_x, y=p95_base, mode='lines', line=dict(width=0), showlegend=False))
fig_base.add_trace(go.Scatter(x=ejes_x, y=p5_base, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(207, 134, 81, 0.1)', name='Rango 90% (Escenario Actual)'))
fig_base.add_trace(go.Scatter(x=ejes_x, y=p75_base, mode='lines', line=dict(width=0), showlegend=False))
fig_base.add_trace(go.Scatter(x=ejes_x, y=p25_base, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(207, 134, 81, 0.25)', name='Rango 50% (Escenario Actual)'))
fig_base.add_trace(go.Scatter(x=ejes_x, y=p50_base, mode='lines+markers', line=dict(color='#CF8651', width=3), name=f'Mediana P50 ({escenario})'))
fig_base.add_hline(y=META_CAJA_MONTO, line_dash="dash", line_color="#D32F2F", annotation_text="Línea de Meta (20 Días)", annotation_position="top left")
fig_base.update_layout(yaxis_title="Disponibilidad de Caja (COP Millones)", hovermode="x unified", template="plotly_white", height=400, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_base, use_container_width=True)


# -----------------------------------------------------------------------------
# 7. BLOQUE 2: ESTRATEGIAS FINANCIERAS Y RESULTADOS
# -----------------------------------------------------------------------------
st.markdown("<div class='section-title'>Módulo 2: Simulador de Impacto de Estrategias Financieras</div>", unsafe_allow_html=True)

st.markdown("""
<div class='instruction-box'>
    <b>Instructivo de Intervención Estratégica</b><br><br>
    Las siguientes 5 estrategias han sido diseñadas para abordar la concentración del riesgo en clientes de alto volumen y tardío pago. Cada estrategia seleccionada alterará matemáticamente el escenario definido en el <b>Módulo 1</b>: algunas liberarán liquidez acelerando los días de cartera (DSO), mientras que otras implicarán un costo asociado (incremento en gastos financieros/administrativos). <br><br>
    <b>Seleccione las estrategias en las casillas correspondientes para observar el recálculo dinámico y la liberación de liquidez en el panel de KPIs inferior.</b>
</div>
""", unsafe_allow_html=True)

col_strat1, col_strat2 = st.columns(2)

with col_strat1:
    st.markdown("<div class='strategy-desc-box'><b>1. Abono Mínimo Innegociable (Cobro cada 8 días)</b><br>Exigencia de pagos parciales para fragmentar el riesgo en la línea corporativa. <i>Efecto: Reduce la cartera general (DSO) en 4 días orgánicos.</i></div>", unsafe_allow_html=True)
    s1 = st.checkbox("Aplicar Estrategia 1")
    
    st.markdown("<div class='strategy-desc-box'><b>2. Distribución de Capacidad (Sustitución de Contrato)</b><br>Atomizar operaciones hacia múltiples clientes medianos en lugar de uno grande. <i>Efecto: Aumenta crecimiento de ventas 1.0% y acelera recaudo (DSO -2 días).</i></div>", unsafe_allow_html=True)
    s2 = st.checkbox("Aplicar Estrategia 2")
    
    st.markdown("<div class='strategy-desc-box'><b>3. Scoring de Riesgo Crediticio</b><br>Restricción de servicios a clientes en cuartil de mora alta sin garantías reales. <i>Efecto: Mejora la calidad de la cartera reduciendo el DSO en 3 días.</i></div>", unsafe_allow_html=True)
    s3 = st.checkbox("Aplicar Estrategia 3")

with col_strat2:
    st.markdown("<div class='strategy-desc-box'><b>4. Cobro Bipartito B2B (Anticipos)</b><br>Exigencia de anticipo operativo (30-40%) para nuevas renovaciones corporativas. <i>Efecto: Impacto profundo en liquidez inicial (DSO -10 días).</i></div>", unsafe_allow_html=True)
    s4 = st.checkbox("Aplicar Estrategia 4")
    
    st.markdown("<div class='strategy-desc-box'><b>5. Descuentos Dinámicos por Pronto Pago</b><br>Sustitución del interés moratorio por incentivos decrecientes. <i>Efecto: Acelera pagos (DSO -5 días), asumiendo un costo financiero extra (Admin +0.3%).</i></div>", unsafe_allow_html=True)
    s5 = st.checkbox("Aplicar Estrategia 5")

# Lógica de alteración de variables basada en el escenario seleccionado en el Módulo 1
dso_mod, dpo_mod, cd_mod, adm_mod, g_mod = dso_base, dpo_base, cd_base, adm_base, g_base

if s1: dso_mod -= 4.0
if s2: g_mod += 0.010; dso_mod -= 2.0
if s3: dso_mod -= 3.0
if s4: dso_mod -= 10.0
if s5: dso_mod -= 5.0; adm_mod += 0.003

dso_mod = max(1.0, dso_mod) 

# Ejecución Optimizada
v_mod_arr, cash_mod_arr, ann_mod_arr, c_mod_arr, a_mod_arr = run_monte_carlo(simulaciones, g_mod, cd_mod, adm_mod, dso_mod, dpo_mod)
caja_mod_dic_p50 = np.percentile(cash_mod_arr[:, -1], 50)

# Deltas y nuevos cálculos comparativos
diferencia_caja = caja_mod_dic_p50 - caja_base_dic_p50
prob_meta_mod = (cash_mod_arr[:, -1] >= META_CAJA_MONTO).mean() * 100
dias_caja_logrados = caja_mod_dic_p50 / DAILY_OUT
costo_integrado_mod = (c_mod_arr.mean() + a_mod_arr.mean()) / v_mod_arr.mean() * 100
diferencia_costo = costo_integrado_mod - costo_integrado_base

st.markdown("<div class='section-title'>Resultados Comparativos de las Estrategias Aplicadas</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card success">
        <div class="kpi-title">Impacto Directo: Efectivo Liberado</div>
        <div class="kpi-value">${diferencia_caja:,.1f} M</div>
        <div class="kpi-desc">Diferencia neta vs. el escenario {escenario}.</div>
    </div>
    <div class="kpi-card {'success' if prob_meta_mod >= 50 else 'alert'}">
        <div class="kpi-title">Nueva Prob. Cumplimiento (20 Días)</div>
        <div class="kpi-value">{prob_meta_mod:.1f}%</div>
        <div class="kpi-desc">Certeza estadística con la estrategia actual.</div>
    </div>
    <div class="kpi-card neutral">
        <div class="kpi-title">Nueva Equivalencia de Caja Lograda</div>
        <div class="kpi-value">{dias_caja_logrados:.1f} Días</div>
        <div class="kpi-desc">Frente a la meta estricta de {META_DIAS} días.</div>
    </div>
    <div class="kpi-card alert">
        <div class="kpi-title">Nuevo Perfil de Cartera (DSO)</div>
        <div class="kpi-value">{dso_mod:.1f} Días</div>
        <div class="kpi-desc">Modificado desde los {dso_base:.1f} días base.</div>
    </div>
    <div class="kpi-card {'success' if diferencia_costo <= 0 else 'risk'}">
        <div class="kpi-title">Nueva Carga Operativa Integrada</div>
        <div class="kpi-value">{costo_integrado_mod:.1f}%</div>
        <div class="kpi-desc">Variación del {diferencia_costo:+.2f}% frente a la base.</div>
    </div>
</div>
""", unsafe_allow_html=True)
