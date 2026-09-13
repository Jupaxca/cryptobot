import ccxt
import pandas as pd
import numpy as np
import os
import json
import requests
from datetime import datetime, timezone

# ==========================================================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================================================
NUCLEO_CONSERVADOR = ['BTC/USD', 'ETH/USD']
TEMPORALIDAD_NUCLEO = '1d'

NIVEL_CRECIMIENTO = ['SOL/USD', 'LINK/USD', 'AVAX/USD']
TEMPORALIDAD_CRECIMIENTO = '1d'

TEMPORALIDAD_SATELITE = '4h'
SATELITE_RSI_ENTRADA = 25
SATELITE_ADX_MAX = 20
SATELITE_ATR_SL_MULT = 2.0
SATELITE_ATR_TP_MULT = 4.0

VELAS_ANALISIS = 300
VELAS_ESCUDO_BTC = 80
TEMPORALIDAD_ESCUDO = '1d'

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

ESTADO_PATH = 'bot_estado.json'
SOLO_ALERTAR_CAMBIOS = True

CAPITAL_REFERENCIA_USD = 90  # Equivale a 370.000 COP en USD para los pares
RIESGO_POR_TRADE = 0.01      # 1% de riesgo por trade
HORIZONTE_VALIDACION = 14

# ==========================================================================
# 2. DESCARGA Y PROCESAMIENTO
# ==========================================================================
def descargar_velas_cerradas(exchange, simbolo, temporalidad, limit):
    try:
        tf_ms = exchange.parse_timeframe(temporalidad) * 1000
        velas = exchange.fetch_ohlcv(simbolo, timeframe=temporalidad, limit=limit + 1)
        if not velas: return velas
        if velas[-1][0] + tf_ms > exchange.milliseconds():
            velas = velas[:-1]
        return velas[-limit:] if len(velas) > limit else velas
    except Exception: return None

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], 100)

def calcular_bollinger_bands(series, period=20, std_dev=2):
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return middle + (std * std_dev), middle, middle - (std * std_dev)

def calcular_atr(df, period=14):
    tr = pd.concat([
        df['maximo'] - df['minimo'],
        (df['maximo'] - df['cierre'].shift(1)).abs(),
        (df['minimo'] - df['cierre'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calcular_adx(df, period=14):
    up_move = df['maximo'] - df['maximo'].shift(1)
    down_move = df['minimo'].shift(1) - df['minimo']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr = calcular_atr(df, period)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.replace([np.inf, -np.inf], np.nan).rolling(period).mean()

def detectar_patrones_smc_hist(df):
    """ Separa la lógica histórica (sin look-ahead) de la señal en vivo """
    senales_alcistas = np.zeros(len(df), dtype=bool)
    senales_bajistas = np.zeros(len(df), dtype=bool)
    ultima_senal = None
    
    if len(df) < 20: return ultima_senal, senales_alcistas, senales_bajistas

    for i in range(15, len(df)):
        # Verificamos si en la vela 'i' o en su entorno cercano ocurrió un FVG con mitigación local
        # Usamos una ventana estricta para el histórico sin mirar el final absoluto del DataFrame
        if i < len(df) - 1:
            if df['minimo'].iloc[i] > df['maximo'].iloc[i-2]:
                if df['minimo'].iloc[i] and df['cierre'].iloc[i] >= df['maximo'].iloc[i-2]:
                    min_sweep = df['minimo'].iloc[max(0, i-15):i].min()
                    if df['minimo'].iloc[max(0, i-4):i].min() == min_sweep:
                        senales_alcistas[i] = True
                        
            if df['maximo'].iloc[i] < df['minimo'].iloc[i-2]:
                if df['maximo'].iloc[i] and df['cierre'].iloc[i] <= df['minimo'].iloc[i-2]:
                    max_sweep = df['maximo'].iloc[max(0, i-15):i].max()
                    if df['maximo'].iloc[max(0, i-4):i].max() == max_sweep:
                        senales_bajistas[i] = True

    # Señal en vivo evaluada estrictamente en la última vela cerrada
    i_live = len(df) - 2
    if i_live >= 15:
        if df['minimo'].iloc[i_live] > df['maximo'].iloc[i_live-2]:
            min_sweep_live = df['minimo'].iloc[i_live-15:i_live].min()
            if df['minimo'].iloc[-1] <= df['minimo'].iloc[i_live] and df['cierre'].iloc[-1] >= df['maximo'].iloc[i_live-2]:
                if df['minimo'].iloc[i_live-4:i_live].min() == min_sweep_live:
                    sl = min_sweep_live * 0.995
                    tp = df['maximo'].iloc[-20:].max()
                    ultima_senal = {"tipo": "SMC_ALCISTA", "sl": sl, "tp": tp}
                    
        if df['maximo'].iloc[i_live] < df['minimo'].iloc[i_live-2]:
            max_sweep_live = df['maximo'].iloc[i_live-15:i_live].max()
            if df['maximo'].iloc[-1] >= df['maximo'].iloc[i_live] and df['cierre'].iloc[-1] <= df['minimo'].iloc[i_live-2]:
                if df['maximo'].iloc[i_live-4:i_live].max() == max_sweep_live:
                    sl = max_sweep_live * 1.005
                    tp = df['minimo'].iloc[-20:].min()
                    ultima_senal = {"tipo": "SMC_BAJISTA", "sl": sl, "tp": tp}

    return ultima_senal, senales_alcistas, senales_bajistas

def validar_senal_historica(df, condicion_activa, horizonte=HORIZONTE_VALIDACION):
    disparos = np.where(condicion_activa)[0]
    disparos = disparos[disparos < len(df) - horizonte]
    if len(disparos) < 5:
        return {'n_casos': len(disparos), 'win_rate': None, 'retorno_promedio': None, 'suficiente': False}
    retornos = np.array([(df.iloc[i + horizonte]['cierre'] / df.iloc[i]['cierre'] - 1) * 100 for i in disparos])
    return {
        'n_casos': len(disparos), 'win_rate': (retornos > 0).mean() * 100,
        'retorno_promedio': retornos.mean(), 'suficiente': len(disparos) >= 15
    }

def es_mercado_spot_valido(exchange, simbolo):
    mercado = exchange.markets.get(simbolo)
    return mercado and mercado.get('spot', False) is True and mercado.get('type', 'spot') == 'spot'

# ==========================================================================
# 3. TELEGRAM Y ESTADO
# ==========================================================================
def enviar_alerta_telegram(mensaje):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                          json={"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}, timeout=15)
        except Exception as e:
            print(f"⚠️ Error enviando Telegram: {e}")

def cargar_estado():
    if os.path.exists(ESTADO_PATH):
        try:
            with open(ESTADO_PATH, 'r') as f: return json.load(f)
        except Exception: pass
    return {}

def guardar_estado(estado):
    with open(ESTADO_PATH, 'w') as f: json.dump(estado, f, indent=2)

# ==========================================================================
# 4. ANÁLISIS DE ACTIVOS
# ==========================================================================
def analizar_activo_largo_plazo(exchange, simbolo, temporalidad, descuento_pct, rsi_compra, rsi_venta, sobreprecio_pct, es_crecimiento=False):
    velas = descargar_velas_cerradas(exchange, simbolo, temporalidad, VELAS_ANALISIS)
    if not velas or len(velas) < 60: return None

    df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
    df['RSI'], df['Media_30'], df['ATR'] = calcular_rsi(df['cierre'], 14), df['cierre'].rolling(30).mean(), calcular_atr(df, 14)
    df = df.dropna().reset_index(drop=True)
    
    precio, rsi, media_30, atr = df.iloc[-1]['cierre'], df.iloc[-1]['RSI'], df.iloc[-1]['Media_30'], df.iloc[-1]['ATR']
    condicion_descuento, condicion_sobrecompra = (df['cierre'] <= df['Media_30'] * (1 - descuento_pct)) | (df['RSI'] < rsi_compra), df['RSI'] >= rsi_venta
    prefijo = "⚠️ *(Alta Volatilidad):* " if es_crecimiento else ""

    if precio <= (media_30 * (1 - descuento_pct)) or rsi < rsi_compra:
        accion, etiqueta, validacion = "COMPRAR", f"{prefijo}🟢 *COMPRAR:* Activo en descuento.", validar_senal_historica(df, condicion_descuento.values)
    elif rsi >= rsi_venta:
        accion, etiqueta, validacion = "EVALUAR_VENTA", f"{prefijo}🟡 *TOMAR BENEFICIOS:* Euforia extrema. Vender fracción.", validar_senal_historica(df, condicion_sobrecompra.values)
    elif precio > (media_30 * (1 + sobreprecio_pct)):
        accion, etiqueta, validacion = "ESPERAR", f"{prefijo}🔴 *ESPERAR:* Precio elevado.", None
    else:
        accion, etiqueta, validacion = "NEUTRO", f"{prefijo}⚪ *ZONA NEUTRA:* Mercado estable.", None

    # Blindaje matemático en el denominador para evitar divisiones por cero o valores absurdos
    denominador_atr = max(atr * 2.5, 0.0001)
    sugerencia_tamano = CAPITAL_REFERENCIA_USD * RIESGO_POR_TRADE / denominador_atr

    return {'simbolo': simbolo, 'precio': precio, 'rsi': rsi, 'media_30': media_30, 'accion': accion, 'etiqueta': etiqueta, 'validacion': validacion, 'sugerencia_tamano': sugerencia_tamano, 'atr': atr}

# ==========================================================================
# 5. BOT MAESTRO
# ==========================================================================
def ejecutar_bot_maestro():
    exchange = ccxt.kraken({'enableRateLimit': True, 'timeout': 30000})
    exchange.load_markets()
    estado_anterior, estado_nuevo = cargar_estado(), {}
    alertas_nucleo, alertas_crecimiento, alertas_satelite = [], [], []

    velas_btc = descargar_velas_cerradas(exchange, 'BTC/USD', TEMPORALIDAD_ESCUDO, VELAS_ESCUDO_BTC)
    btc_saludable = False
    if velas_btc:
        df_btc = pd.DataFrame(velas_btc, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
        btc_saludable = df_btc.iloc[-1]['cierre'] >= df_btc['cierre'].rolling(50).mean().iloc[-1]

    # --- Núcleo y Crecimiento ---
    for s in NUCLEO_CONSERVADOR:
        if r := analizar_activo_largo_plazo(exchange, s, TEMPORALIDAD_NUCLEO, 0.04, 40, 75, 0.05, False):
            alertas_nucleo.append(r)
            estado_nuevo[s] = r['accion']

    for s in NIVEL_CRECIMIENTO:
        if r := analizar_activo_largo_plazo(exchange, s, TEMPORALIDAD_CRECIMIENTO, 0.07, 35, 80, 0.08, True):
            alertas_crecimiento.append(r)
            estado_nuevo[s] = r['accion']

    # --- Satélite (Protegido con try/except para no tumbar el reporte principal si falla) ---
    if btc_saludable:
        try:
            tickers = exchange.fetch_tickers()
            excluidos = set(NUCLEO_CONSERVADOR) | set(NIVEL_CRECIMIENTO)
            candidatas = sorted([s for s, t in tickers.items() if '/USD' in s and s not in excluidos and 'USDT' not in s and es_mercado_spot_valido(exchange, s) and t.get('quoteVolume', 0) > 1000000], key=lambda s: tickers[s].get('quoteVolume', 0), reverse=True)[:10]

            for s in candidatas:
                velas = descargar_velas_cerradas(exchange, s, TEMPORALIDAD_SATELITE, VELAS_ANALISIS)
                if not velas or len(velas) < 60: continue
                
                df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
                df['RSI'], _, df['BB_Lower'] = calcular_rsi(df['cierre'], 14), None, calcular_bollinger_bands(df['cierre'])[2]
                df['Vol_Medio'], df['ATR'], df['ADX'], df['EMA_TENDENCIA'] = df['volumen'].rolling(20).mean(), calcular_atr(df, 14), calcular_adx(df, 14), df['cierre'].ewm(span=200, adjust=False).mean()
                df = df.dropna().reset_index(drop=True)
                ultima = df.iloc[-1]
                
                patron_smc, hist_alc, hist_baj = detectar_patrones_smc_hist(df)
                cuantitativo_ok = (ultima['RSI'] <= SATELITE_RSI_ENTRADA and ultima['cierre'] <= ultima['BB_Lower'] * 1.01 and ultima['volumen'] >= ultima['Vol_Medio'] * 0.7 and ultima['ADX'] < SATELITE_ADX_MAX and ultima['cierre'] > ultima['EMA_TENDENCIA'])

                # Prioridad 1: Cuantitativo (Más validado y seguro)
                if cuantitativo_ok:
                    sl, tp = ultima['cierre'] - (SATELITE_ATR_SL_MULT * ultima['ATR']), ultima['cierre'] + (SATELITE_ATR_TP_MULT * ultima['ATR'])
                    val = validar_senal_historica(df, (df['RSI'] <= SATELITE_RSI_ENTRADA) & (df['cierre'] <= df['BB_Lower'] * 1.01))
                    denominador_riesgo = max(abs(ultima['cierre'] - sl), 0.0001)
                    sugerencia_tamano = CAPITAL_REFERENCIA_USD * RIESGO_POR_TRADE / denominador_riesgo
                    
                    alertas_satelite.append({'simbolo': s, 'precio': ultima['cierre'], 'rsi': ultima['RSI'], 'tipo_alerta': 'CUANTITATIVO_ALCISTA', 'tp': tp, 'sl': sl, 'sugerencia_tamano': sugerencia_tamano, 'validacion': val})
                    estado_nuevo[s] = 'CUANTITATIVO_ALCISTA'
                # Prioridad 2: SMC 
                elif patron_smc:
                    val_smc = validar_senal_historica(df, hist_alc if patron_smc['tipo'] == "SMC_ALCISTA" else hist_baj)
                    denominador_riesgo = max(abs(ultima['cierre'] - patron_smc['sl']), 0.0001)
                    sugerencia_tamano = CAPITAL_REFERENCIA_USD * RIESGO_POR_TRADE / denominador_riesgo
                    
                    alertas_satelite.append({'simbolo': s, 'precio': ultima['cierre'], 'rsi': ultima['RSI'], 'tipo_alerta': patron_smc['tipo'], 'tp': patron_smc['tp'], 'sl': patron_smc['sl'], 'sugerencia_tamano': sugerencia_tamano, 'validacion': val_smc})
                    estado_nuevo[s] = patron_smc['tipo']
        except Exception as e:
            print(f"⚠️ Error procesando el bloque satélite: {e}")

    hubo_cambio = lambda s, a: estado_anterior.get(s) != a
    filtrar = lambda l, c: [a for a in l if not SOLO_ALERTAR_CAMBIOS or hubo_cambio(a['simbolo'], a.get('accion', c))]
    
    n_env, c_env, s_env = filtrar(alertas_nucleo, None), filtrar(alertas_crecimiento, None), [a for a in alertas_satelite if not SOLO_ALERTAR_CAMBIOS or hubo_cambio(a['simbolo'], a['tipo_alerta'])]

    if n_env or c_env or s_env:
        msj = f"🚨 *REPORTE DE INVERSIÓN (EXCEL & SMC)* — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        if n_env:
            msj += "🛡️ *NÚCLEO CONSERVADOR*\n" + "".join([f"• *{o['simbolo']}* | `${o['precio']:,.2f}`\n  {o['etiqueta']}\n" for o in n_env]) + "\n"
        if c_env:
            msj += "🚀 *CRECIMIENTO Y ALTO POTENCIAL*\n" + "".join([f"• *{o['simbolo']}* | `${o['precio']:,.2f}`\n  {o['etiqueta']}\n" for o in c_env]) + "\n"
        if s_env:
            msj += f"🎯 *RADAR DE ALTO RIESGO ({TEMPORALIDAD_SATELITE})*\n"
            for o in s_env:
                msj += f"{'🟢' if 'ALCISTA' in o['tipo_alerta'] else '🔴'} *{o['tipo_alerta'].replace('_', ' ')}* | *{o['simbolo']}*\n  Entrada: `${o['precio']:,.4f}`\n  🎯 TP: `${o['tp']:,.4f}` | 🛑 SL: `${o['sl']:,.4f}`\n  📏 Tamaño sugerido: `{o['sugerencia_tamano']:,.2f}` UND\n"
                if o['validacion'] and o['validacion']['suficiente']:
                    msj += f"  📊 Validado ({o['validacion']['n_casos']} casos): Win Rate {o['validacion']['win_rate']:.0f}%, Promedio {o['validacion']['retorno_promedio']:+.1f}%\n"
                msj += "\n"
        msj += "_⚠️ Herramienta de apoyo analítico. No constituye asesoría financiera personalizada. Verifique siempre antes de operar._"
        enviar_alerta_telegram(msj)
    
    guardar_estado(estado_nuevo)

if __name__ == '__main__':
    ejecutar_bot_maestro()
