import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Dashboard IA Trading", layout="wide")
st.title("🤖 Panel de Control: IA de Trading")

# Función para cargar datos en tiempo real sin caché prolongado
@st.cache_data(ttl=60) # Recarga los datos cada 60 segundos automáticamente
def cargar_datos():
    try:
        df = pd.read_csv('memoria_predicciones.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = cargar_datos()

if df.empty:
    st.warning("Aún no hay datos en la memoria. Deja que el bot ejecute algunas alertas.")
else:
    # 1. MÉTRICAS CLAVE (KPIs)
    st.header("📊 Rendimiento Global")
    
    # Filtrar solo los trades auditados (que ya tienen resultado real)
    df_auditado = df.dropna(subset=['exito_real']).copy()
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_alertas = len(df)
    alertas_completadas = len(df_auditado)
    
    if alertas_completadas > 0:
        win_rate = (df_auditado['exito_real'].sum() / alertas_completadas) * 100
        racha_actual = df_auditado.tail(10)['exito_real'].mean() * 100
    else:
        win_rate = 0
        racha_actual = 0

    col1.metric("Total Alertas Generadas", total_alertas)
    col2.metric("Alertas Auditadas", alertas_completadas)
    col3.metric("Win Rate Histórico", f"{win_rate:.1f}%")
    
    # Mostrar si la racha actual está mejorando o empeorando respecto al histórico
    delta_racha = racha_actual - win_rate if alertas_completadas >= 10 else 0
    col4.metric("Win Rate (Últimos 10)", f"{racha_actual:.1f}%", f"{delta_racha:.1f}%")

    st.divider()

    # 2. GRÁFICOS INTERACTIVOS
    st.header("📈 Análisis de Tendencias")
    
    tab1, tab2 = st.tabs(["Evolución de Aciertos", "Distribución por Moneda"])
    
    with tab1:
        if alertas_completadas > 0:
            # Calcular Win Rate Acumulado en el tiempo
            df_auditado['win_rate_acumulado'] = df_auditado['exito_real'].expanding().mean() * 100
            fig_wr = px.line(df_auditado, x='timestamp', y='win_rate_acumulado', 
                             title="Evolución del Win Rate de la IA",
                             markers=True)
            st.plotly_chart(fig_wr, use_container_width=True)
            
    with tab2:
        if alertas_completadas > 0:
            # Aciertos vs Fallos por criptomoneda
            resumen_monedas = df_auditado.groupby('simbolo')['exito_real'].agg(['count', 'mean']).reset_index()
            resumen_monedas.columns = ['Moneda', 'Trades', 'Win Rate']
            resumen_monedas['Win Rate'] = resumen_monedas['Win Rate'] * 100
            
            fig_bar = px.bar(resumen_monedas, x='Moneda', y='Win Rate', 
                             color='Trades', title="Rendimiento por Activo")
            fig_bar.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Punto de Equilibrio (50%)")
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # 3. TABLA DE DATOS EN VIVO
    st.header("📋 Historial Detallado")
    
    # Formatear la tabla para que sea más legible
    df_mostrar = df.sort_values(by='timestamp', ascending=False).copy()
    df_mostrar['Prob. Alza (%)'] = df_mostrar['prob_subida_predicha'].round(2)
    df_mostrar['Resultado'] = df_mostrar['exito_real'].map({1.0: '✅ Éxito', 0.0: '❌ Fallo', pd.NA: '⏳ Pendiente'})
    
    columnas_visibles = ['timestamp', 'simbolo', 'precio_entrada', 'RSI', 'Prob. Alza (%)', 'Resultado']
    st.dataframe(df_mostrar[columnas_visibles], use_container_width=True, hide_index=True)
