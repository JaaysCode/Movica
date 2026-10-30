# -----------------------------------------------------------------------------
# 0. INSTALACIÓN DE LIBRERÍAS E IMPORTACIÓN DE MÓDULOS
# Ejecutar en terminal: pip install streamlit numpy pandas plotly
# -----------------------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CORPORATIVOS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Movica | Panel de Tesorería", layout="wide")

# Estilos CSS basados en la paleta del reporte corporativo de Servitravel
st.markdown("""
    <style>
    .kpi-container { display: flex; gap: 15px; margin-bottom: 25px; }
    .kpi-card {
        flex: 1;
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #186663; /* Teal corporativo */
    }
    .kpi-card.alert { border-left: 5px solid #CF8651; /* Naranja/Terracota corporativo */ }
    .kpi-card.risk { border-left: 5px solid #D32F2F; /* Rojo alerta */ }
    .kpi-card.neutral { border-left: 5px solid #637397; /* Gris pizarra corporativo */ }
    .kpi-title { color: #637397; font-size: 12px; font-weight: 700; text-transform: uppercase; margin-bottom: 5px; }
    .kpi-value { color: #1E293B; font-size: 28px; font-weight: bold; margin: 0; }
    .kpi-desc { color: #9CA3AF; font-size: 11px; margin-top: 5px; }
    .strategy-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .strategy-title { font-weight: bold; color: #186663; font-size: 14px; margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# AUTENTICACIÓN Y SEGURIDAD
# -----------------------------------------------------------------------------
def check_password():
    def password_entered():
        if st.session_state["pwd_input"] == st.secrets["password"]:
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
# 2. INGESTA DE DATOS HISTÓRICOS Y CONSTANTES
# -----------------------------------------------------------------------------
MESES = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago']
V = np.array([2339.690875, 2512.919954, 2640.693165, 2690.628864, 2703.839974, 2427.660012, 2355.187919, 2678.096])
PROP = np.array([362.386894, 408.541817, 406.194476, 389.132349, 414.776652, 332.361208, 211.992726, 461.129])
TERC = np.array([501.641998, 707.374951, 704.719649, 692.311182, 787.160413, 922.691001, 831.263665, 877.464])
PROV = np.array([1147.784008, 1101.584171, 1203.978169, 1272.752021, 1190.998477, 1000.378140, 1007.470092, 1045.193])
ADM = np.array([188.618102, 190.775015, 235.996692, 220.610575, 212.558149, 209.998828, 217.002004, 201.192])
FAM = np.array([.975054, .949047, .553104, .557104, 0, .610540, .968841, 0])
OI  = np.array([26.164481, 29.156727, 31.007139, 25.314360, 29.824055, 64.081929, 27.888190, 25.814])
OG  = np.array([22.177279, 18.080754, 22.085206, 23.863840, 27.428479, 23.588643, 23.517142, 24.665])
CF  = np.array([9.538267, 9.772962, 10.625225, 9.339117, 10.172358, 10.706427, 11.384398, 13.690])

CD = PROP + TERC + PROV
RAI = V - CD - ADM - FAM + OI - OG - CF

BAL_AGO = dict(cash=161.730353, cxc=3697.300413, cxp=246.6)
TAX, DEP = 0.35, 9.0
T = 4  # Sep-Dic

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
# 3. MOTOR DE SIMULACIÓN MONTE CARLO
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
# 4. INTERFAZ DE USUARIO (DASHBOARD)
# -----------------------------------------------------------------------------
st.title("Movica | Panel Estratégico de Gestión de Liquidez")
st.markdown("Proyección estocástica orientada al aseguramiento de reservas operativas y mitigación de riesgo de impago.")

# Panel lateral para parámetros
with st.sidebar:
    st.header("Parámetros de Simulación")
    escenario = st.selectbox("Escenario Macroeconómico", ["Base", "Pesimista", "Optimista", "Personalizado"])
    
    if escenario == "Base":
        cd_val = st.slider("Costo Directo (% Vtas)", 88.0, 89.0, 88.5, 0.1)
        adm_val = st.slider("Gastos Admin (% Vtas)", 8.0, 8.5, 8.24, 0.01)
        dso_val = st.slider("Días Cartera (DSO)", 21.0, 22.0, 21.5, 0.5)
        dpo_val = st.slider("Días Proveedores (DPO)", 14.0, 16.0, 15.0, 0.5)
        g_val = st.slider("Crecimiento Mensual (%)", 0.0, 0.5, 0.2, 0.1)
    elif escenario == "Pesimista":
        cd_val = st.slider("Costo Directo (% Vtas)", 91.6, 93.0, 92.3, 0.1)
        adm_val = st.slider("Gastos Admin (% Vtas)", 9.0, 9.5, 9.21, 0.01)
        dso_val = st.slider("Días Cartera (DSO)", 25.0, 30.0, 27.5, 0.5)
        dpo_val = st.slider("Días Proveedores (DPO)", 10.0, 12.0, 11.0, 0.5)
        g_val = st.slider("Crecimiento Mensual (%)", -2.0, -1.0, -1.5, 0.1)
    elif escenario == "Optimista":
        cd_val = st.slider("Costo Directo (% Vtas)", 84.0, 85.0, 84.5, 0.1)
        adm_val = st.slider("Gastos Admin (% Vtas)", 7.0, 7.6, 7.51, 0.01)
        dso_val = st.slider("Días Cartera (DSO)", 18.0, 19.0, 18.5, 0.5)
        dpo_val = st.slider("Días Proveedores (DPO)", 22.0, 25.0, 23.5, 0.5)
        g_val = st.slider("Crecimiento Mensual (%)", 1.5, 2.5, 2.0, 0.1)
    else:
        cd_val = st.slider("Costo Directo (% Vtas)", 80.0, 95.0, 88.5, 0.1)
        adm_val = st.slider("Gastos Admin (% Vtas)", 5.0, 12.0, 8.24, 0.1)
        dso_val = st.slider("Días Cartera (DSO)", 10.0, 60.0, 21.5, 1.0)
        dpo_val = st.slider("Días Proveedores (DPO)", 5.0, 30.0, 15.0, 1.0)
        g_val = st.slider("Crecimiento Mensual (%)", -5.0, 5.0, 0.2, 0.1)

    simulaciones = st.selectbox("Iteraciones (Monte Carlo)", [5000, 10000, 20000], index=1)

# Ejecución del modelo
v_mc, cash_mc, ann_mc, c_mc, a_mc = run_monte_carlo(simulaciones, g_val / 100, cd_val / 100, adm_val / 100, dso_val, dpo_val)

# Cálculos para Tarjetas KPI
prob_meta = (cash_mc[:, -1] >= META_CAJA_MONTO).mean() * 100
costos_totales_proyectados = (c_mc.mean() + a_mc.mean()) / v_mc.mean() * 100
cash_p50 = np.percentile(cash_mc[:, -1], 50)
dias_caja_proyectados = cash_p50 / DAILY_OUT

# Renderizado de Tarjetas KPI
st.markdown("""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-title">Monto Objetivo (20 Días)</div>
        <div class="kpi-value">${:,.1f} M</div>
        <div class="kpi-desc">Equivalente a 20 días de operación a $85.3M/día.</div>
    </div>
    <div class="kpi-card neutral">
        <div class="kpi-title">Probabilidad de Cumplimiento</div>
        <div class="kpi-value">{:.1f}%</div>
        <div class="kpi-desc">Certeza estadística de alcanzar la meta a diciembre.</div>
    </div>
    <div class="kpi-card alert">
        <div class="kpi-title">Equivalencia en Cartera (DSO)</div>
        <div class="kpi-value">{:.1f} Días</div>
        <div class="kpi-desc">Días de recaudo simulados (Histórico: 44.6 días).</div>
    </div>
    <div class="kpi-card risk">
        <div class="kpi-title">Carga de Costos Supuestos</div>
        <div class="kpi-value">{:.1f}%</div>
        <div class="kpi-desc">Suma ponderada de admin, flota propia y tercerización.</div>
    </div>
</div>
""".format(META_CAJA_MONTO, prob_meta, dso_val, costos_totales_proyectados), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. VISUALIZACIÓN GRÁFICA Y ESTRATEGIAS
# -----------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Trayectoria de Caja Proyectada")
    percentiles = np.percentile(cash_mc, [5, 25, 50, 75, 95], axis=0)
    ejes_x = ['Ago (Real)', 'Sep', 'Oct', 'Nov', 'Dic']
    
    p5 = np.insert(percentiles[0], 0, BAL_AGO['cash'])
    p25 = np.insert(percentiles[1], 0, BAL_AGO['cash'])
    p50 = np.insert(percentiles[2], 0, BAL_AGO['cash'])
    p75 = np.insert(percentiles[3], 0, BAL_AGO['cash'])
    p95 = np.insert(percentiles[4], 0, BAL_AGO['cash'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ejes_x, y=p95, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=ejes_x, y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(24, 102, 99, 0.1)', name='Rango 90%'))
    fig.add_trace(go.Scatter(x=ejes_x, y=p75, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=ejes_x, y=p25, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(24, 102, 99, 0.25)', name='Rango 50%'))
    fig.add_trace(go.Scatter(x=ejes_x, y=p50, mode='lines+markers', line=dict(color='#186663', width=3), name='Mediana (P50)'))
    
    fig.add_hline(y=META_CAJA_MONTO, line_dash="dash", line_color="#D32F2F", annotation_text=f"Meta: {META_DIAS} Días", annotation_position="top left")
    
    fig.update_layout(
        xaxis_title="Período", yaxis_title="Liquidez (COP Millones)",
        hovermode="x unified", template="plotly_white", height=450, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Manual de Ejecución Estratégica")
    st.markdown(f"""
    <div style="height: 450px; overflow-y: auto; padding-right: 10px;">
        <div class="strategy-box">
            <div class="strategy-title">1. Reingeniería de Pagos (Confirming)</div>
            Extender cuentas por pagar reteniendo efectivo internamente sin afectar la cadena de suministro, mitigando la brecha temporal entre cobro y pago.
        </div>
        <div class="strategy-box">
            <div class="strategy-title">2. Abono Mínimo Innegociable</div>
            Fragmentar pagos altos de clientes de tardío pago mediante contacto cada 8 días. La prórroga se condiciona a un desembolso parcial del 20-30%.
        </div>
        <div class="strategy-box">
            <div class="strategy-title">3. Factoraje Selectivo</div>
            Venta de cartera de los rubros más altos sin recurso, asumiendo el descuento financiero para inyectar liquidez inmediata a la reserva de 20 días.
        </div>
        <div class="strategy-box">
            <div class="strategy-title">4. Optimización Estructural Híbrida</div>
            Consolidar los costos administrativos (800) y de flota/tercerización (1700) bajo un modelo Pay-when-Paid, indexando el pago logístico al recaudo final.
        </div>
        <div class="strategy-box">
            <div class="strategy-title">5. Distribución de Capacidad (Noviembre)</div>
            Reemplazar el contrato principal saliente atomizando el riesgo operativo entre 3 o 4 clientes medianos con políticas de recaudo estrictas de 15 a 20 días.
        </div>
        <div class="strategy-box">
            <div class="strategy-title">6. Scoring de Riesgo Crediticio</div>
            Agrupación bidimensional cruzando volumen de facturación y días de mora. Clientes en el cuartil de riesgo alto requieren garantías reales para operar.
        </div>
        <div class="strategy-box">
            <div class="strategy-title">7. Cobro Bipartito B2B</div>
            Estructuración de contratos nuevos exigiendo anticipos operativos del 30% al 40% para fondear la operación logística directa sin tocar el flujo de caja diario.
        </div>
        <div class="strategy-box">
            <div class="strategy-title">8. Descuentos Dinámicos (Pronto Pago)</div>
            Reemplazar el cobro de intereses moratorios inefectivos por incentivos decrecientes de pago rápido, monetizando la agilidad de los clientes corporativos.
        </div>
        <div class="strategy-box">
            <div class="strategy-title">9. Cash Pooling Restrictivo</div>
            Implementar barrido de saldos diario aislando entre 1 y 2 días operativos hacia el fondo a la vista, hasta lograr progresivamente el piso de ${META_CAJA_MONTO:,.0f}M.
        </div>
        <div class="strategy-box">
            <div class="strategy-title">10. Cobertura de Riesgo Cruzado</div>
            Auditoría de capacidad financiera de proveedores críticos y adquisición de seguro de crédito comercial para cuentas por cobrar con alta exposición.
        </div>
    </div>
    """, unsafe_allow_html=True)
