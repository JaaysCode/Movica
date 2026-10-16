# -----------------------------------------------------------------------------
# 0. IMPORTACIÓN DE LIBRERÍAS
# -----------------------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Movica | Simulador", page_icon="🚌", layout="wide")

# Ocultar el menú de Streamlit, el ícono de GitHub y el pie de página
ocultar_menu = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none;}
    </style>
"""
st.markdown(ocultar_menu, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. FUNCIÓN DE CONTRASEÑA
# -----------------------------------------------------------------------------
def check_password():
    """Retorna True si el usuario ingresa la contraseña correcta."""
    def password_entered():
        if st.session_state["password"] == "Movica2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  #
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown(" Acceso Restringido ")
        st.text_input("Por favor, ingresa la contraseña para acceder al Simulador Financiero:",
                      type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown(" Acceso Restringido ")
        st.text_input("Por favor, ingresa la contraseña para acceder al Simulador Financiero:",
                      type="password", on_change=password_entered, key="password")
        st.error(" Contraseña incorrecta. Intenta nuevamente.")
        return False
    else:
        return True

# -----------------------------------------------------------------------------
# 2. INGESTA DE DATOS HISTÓRICOS Y CONSTANTES (ENERO - AGOSTO 2026)
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

def lerp(a, b, k):
    return a + (b - a) * k / T

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

    m4 = r.sum(1) / v.sum(1)
    ann = (YTD_RAI + r.sum(1)) / (YTD_V + v.sum(1))

    return v, r, cash, m4, ann

# -----------------------------------------------------------------------------
# 3. EJECUCIÓN CONDICIONAL DEL SIMULADOR (STREAMLIT)
# -----------------------------------------------------------------------------
if check_password():
    
    # Encabezado
    st.markdown("""
    <div style="background-color: #1e293b; padding: 15px; border-radius: 8px; color: white; font-family: sans-serif; margin-bottom: 15px;">
        <h2 style="margin: 0; color: #ffffff;"> Movica | Simulador Financiero Interactivo</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabla de Información Estratégica
    st.markdown("""
    <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 15px; margin-bottom: 20px; font-family: sans-serif; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <h4 style="margin-top: 0; color: #1e293b; font-size: 14px;"> Guía de Parametrización Estratégica</h4>
        <p style="font-size: 13px; color: #475569; margin-bottom: 10px;">Para asegurar una toma de decisiones informada y fundamentada en el diagnóstico financiero corporativo, las variables de simulación se encuentran restringidas a los siguientes rangos operativos según el escenario seleccionado:</p>
        <table style="width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; color: #1e293b;">
            <thead>
                <tr style="background-color: #e2e8f0; border-bottom: 2px solid #cbd5e1;">
                    <th style="padding: 8px;">Variable de Control</th>
                    <th style="padding: 8px;">Escenario Base</th>
                    <th style="padding: 8px;">Escenario Optimista</th>
                    <th style="padding: 8px;">Escenario Pesimista</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 8px; font-weight: bold;">Costo Directo (% Vtas)</td>
                    <td style="padding: 8px;">88.0% - 89.0%</td>
                    <td style="padding: 8px;">84.0% - 85.0%</td>
                    <td style="padding: 8px;">91.6% - 93.0%</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 8px; font-weight: bold;">Gastos Admin (% Vtas)</td>
                    <td style="padding: 8px;">8.0% - 8.5%</td>
                    <td style="padding: 8px;">7.0% - 7.6%</td>
                    <td style="padding: 8px;">9.0% - 9.5%</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 8px; font-weight: bold;">Crecimiento Mensual (%)</td>
                    <td style="padding: 8px;">0.0% - 0.5%</td>
                    <td style="padding: 8px;">1.5% - 2.5%</td>
                    <td style="padding: 8px;">-1.0% - -2.0%</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 8px; font-weight: bold;">Días Cartera (DSO)</td>
                    <td style="padding: 8px;">21 - 22 días</td>
                    <td style="padding: 8px;">18 - 19 días</td>
                    <td style="padding: 8px;">25 - 30 días</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Días Proveedores (DPO)</td>
                    <td style="padding: 8px;">14 - 16 días</td>
                    <td style="padding: 8px;">22 - 25 días</td>
                    <td style="padding: 8px;">10 - 12 días</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Panel de Control de Variables (Ajusta los parámetros):**")

    # Controles Interactivos de Streamlit
    escenario = st.selectbox('Escenario Predefinido:', ['Base', 'Pesimista', 'Optimista', 'Personalizado'])

    # Dinámica de Rangos según el escenario
    if escenario == 'Base':
        limites = {'cd': (88.0, 89.0, 88.5), 'adm': (8.0, 8.5, 8.24), 'g': (0.0, 0.5, 0.2), 'dso': (21.0, 22.0, 21.5), 'dpo': (14.0, 16.0, 15.0)}
    elif escenario == 'Pesimista':
        limites = {'cd': (91.6, 93.0, 92.3), 'adm': (9.0, 9.5, 9.21), 'g': (-2.0, -1.0, -1.5), 'dso': (25.0, 30.0, 27.5), 'dpo': (10.0, 12.0, 11.0)}
    elif escenario == 'Optimista':
        limites = {'cd': (84.0, 85.0, 84.5), 'adm': (7.0, 7.6, 7.51), 'g': (1.5, 2.5, 2.0), 'dso': (18.0, 19.0, 18.5), 'dpo': (22.0, 25.0, 23.5)}
    else: # Personalizado
        limites = {'cd': (80.0, 95.0, 88.5), 'adm': (5.0, 12.0, 8.24), 'g': (-5.0, 5.0, 0.2), 'dso': (10.0, 60.0, 21.5), 'dpo': (5.0, 30.0, 15.0)}

    col1, col2 = st.columns(2)
    with col1:
        cd = st.slider('Costo Directo (% Vtas):', min_value=limites['cd'][0], max_value=limites['cd'][1], value=limites['cd'][2], step=0.1)
        g = st.slider('Crecimiento Mensual (%):', min_value=limites['g'][0], max_value=limites['g'][1], value=limites['g'][2], step=0.1)
        dso = st.slider('Días Cartera (DSO):', min_value=limites['dso'][0], max_value=limites['dso'][1], value=limites['dso'][2], step=0.5)
    with col2:
        adm = st.slider('Gastos Admin (% Vtas):', min_value=limites['adm'][0], max_value=limites['adm'][1], value=limites['adm'][2], step=0.01)
        sim = st.selectbox('N° Simulaciones:', [1000, 5000, 10000, 20000], index=2)
        dpo = st.slider('Días Proveedores (DPO):', min_value=limites['dpo'][0], max_value=limites['dpo'][1], value=limites['dpo'][2], step=0.5)

    # 4. EJECUCIÓN Y RENDERIZADO (Se ejecuta automáticamente en Streamlit)
    v_mc, r_mc, cash_mc, m4_mc, ann_mc = run_monte_carlo(sim, g / 100, cd / 100, adm / 100, dso, dpo)

    prob_margen_5 = (ann_mc >= 0.05).mean() * 100
    prob_meta_caja = (cash_mc[:, -1] >= (20 * DAILY_OUT)).mean() * 100
    prob_deficit = (cash_mc.min(1) < 0).mean() * 100
    cash_p50 = np.percentile(cash_mc[:, -1], 50)

    # Tarjetas KPI
    st.markdown(f"""
    <div style="display: flex; gap: 15px; margin-top: 10px; margin-bottom: 20px; font-family: sans-serif;">
        <div style="flex: 1; background: #ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 11px; color: #6c757d; font-weight: bold;">PROB. MARGEN NETO ≥ 5%</div>
            <div style="font-size: 24px; font-weight: bold; color: #1e293b;">{prob_margen_5:.1f}%</div>
            <div style="font-size: 11px; color: {'#10b981' if prob_margen_5 > 50 else '#ef4444'}; font-weight: bold;">Meta Anual Corporativa</div>
        </div>
        <div style="flex: 1; background: #ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #10b981; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 11px; color: #6c757d; font-weight: bold;">PROB. CAJA ≥ 20 DÍAS</div>
            <div style="font-size: 24px; font-weight: bold; color: #1e293b;">{prob_meta_caja:.1f}%</div>
            <div style="font-size: 11px; color: {'#10b981' if prob_meta_caja > 50 else '#ef4444'}; font-weight: bold;">Cobertura ($1,707M)</div>
        </div>
        <div style="flex: 1; background: #ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #ef4444; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 11px; color: #6c757d; font-weight: bold;">PROB. DÉFICIT DE CAJA</div>
            <div style="font-size: 24px; font-weight: bold; color: #1e293b;">{prob_deficit:.1f}%</div>
            <div style="font-size: 11px; color: {'#10b981' if prob_deficit < 5 else '#ef4444'}; font-weight: bold;">Riesgo de Caja &lt; $0</div>
        </div>
        <div style="flex: 1; background: #ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #f59e0b; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 11px; color: #6c757d; font-weight: bold;">CAJA PROYECTADA (DIC P50)</div>
            <div style="font-size: 24px; font-weight: bold; color: #1e293b;">${cash_p50:,.0f}M</div>
            <div style="font-size: 11px; color: #10b981; font-weight: bold;">Millones COP Disponibles</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Gráfico Plotly
    percentiles = np.percentile(cash_mc, [5, 25, 50, 75, 95], axis=0)
    ejes_x = ['Ago (Real)', 'Sep', 'Oct', 'Nov', 'Dic']
    p5 = np.insert(percentiles[0], 0, BAL_AGO['cash'])
    p25 = np.insert(percentiles[1], 0, BAL_AGO['cash'])
    p50 = np.insert(percentiles[2], 0, BAL_AGO['cash'])
    p75 = np.insert(percentiles[3], 0, BAL_AGO['cash'])
    p95 = np.insert(percentiles[4], 0, BAL_AGO['cash'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ejes_x, y=p95, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=ejes_x, y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(31, 119, 180, 0.15)', name='Rango P5 - P95 (90% Confianza)'))
    fig.add_trace(go.Scatter(x=ejes_x, y=p75, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=ejes_x, y=p25, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(31, 119, 180, 0.30)', name='Rango P25 - P75 (50% Confianza)'))
    fig.add_trace(go.Scatter(x=ejes_x, y=p50, mode='lines+markers', line=dict(color='#1f77b4', width=3), name='Mediana (P50)'))
    fig.add_hline(y=(20 * DAILY_OUT), line_dash="dash", line_color="#ef4444", annotation_text="Meta 20 Días de Caja ($1,707M)", annotation_position="top left") 

    fig.update_layout(
        title="Trayectoria de Caja Proyectada — Simulación Monte Carlo",
        xaxis_title="Mes", yaxis_title="Efectivo Disponible (COP Millones)",
        hovermode="x unified", template="plotly_white", height=420, margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Se renderiza en Streamlit
    st.plotly_chart(fig, use_container_width=True)
