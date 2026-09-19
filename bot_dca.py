import ccxt
import pandas as pd
import numpy as np
import os
import json
import requests
import warnings
import time
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ==========================================================================
# 1. CONFIGURACIÓN GENERAL DEL PORTAFOLIO (NARRATIVAS 2026)
# ==========================================================================
NUCLEO_CONSERVADOR = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT']
TEMPORALIDAD_NUCLEO = '1d'

NIVEL_CRECIMIENTO = ['NEAR/USDT', 'TAO/USDT', 'AVAX/USDT', 'RENDER/USDT', 'FET/USDT', 'TIA/USDT', 'ARB/USDT']
TEMPORALIDAD_CRECIMIENTO = '1d'

SEGUIMIENTO_ESTRATEGICO = ['XRP/USDT', 'SUI/USDT', 'APT/USDT', 'ONDO/USDT', 'OM/USDT', 'HNT/USDT', 'FIL/USDT', 'PENDLE/USDT']
TEMPORALIDAD_SEGUIMIENTO = '1d'

ALTO_RIESGO = ['PEPE/USDT', 'WIF/USDT', 'POPCAT/USDT', 'KAS/USDT', 'THETA/USDT', 'ASTR/USDT']
TEMPORALIDAD_ALTO_RIESGO = '1d'

TEMPORALIDAD_SATELITE = '4h'
SATELITE_RSI_ENTRADA = 25
SATELITE_ADX_MAX = 20
SATELITE_ATR_SL_MULT = 2.0
SATELITE_ATR_TP_MULT = 4.0

VELAS_ANALISIS = 1500
VELAS_ESCUDO_BTC = 400
TEMPORALIDAD_ESCUDO = '1d'

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

ESTADO_PATH = 'bot_estado.json'
ARCHIVO_MEMORIA = 'memoria_predicciones.csv'
SOLO_ALERTAR_CAMBIOS = False

CAPITAL_REFERENCIA_USD = 90
RIESGO_POR_TRADE = 0.01
HORIZONTE_VALIDACION = 14
COMISION_EXCHANGE = 0.001
SLIPPAGE = 0.0005

MIN_CASOS_RIESGO_DINAMICO = 20

# ==========================================================================
# 2. PROCESAMIENTO TÉCNICO Y MATEMÁTICAS
# ==========================================================================
def descargar_velas_cerradas(exchange, simbolo, temporalidad, limit):
    try:
        velas = exchange.fetch_ohlcv(simbolo, timeframe=temporalidad, limit=limit)
        return velas
    except Exception:
        return None


def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], 100)


def calcular_vwma(df, period=30):
    return (df['cierre'] * df['volumen']).rolling(period).sum() / df['volumen'].rolling(period).sum()


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


def calcular_hurst_vectorizado(series, window=100, max_lag=20):
    hursts = []
    vals = series.values
    for i in range(len(vals)):
        if i < window:
            hursts.append(0.5)
            continue
        sub = vals[i - window:i]
        try:
            lags = range(2, min(max_lag, window // 2))
            tau = [np.sqrt(np.std(sub[l:] - sub[:-l])) for l in lags]
            poly = np.polyfit(np.log(lags), np.log(np.maximum(tau, 1e-8)), 1)
            hursts.append(poly[0] * 2.0)
        except Exception:
            hursts.append(0.5)
    return pd.Series(hursts, index=series.index)


def calcular_regresion_rolling(series, window=20):
    slopes, r2s = [], []
    vals = series.values
    for i in range(len(vals)):
        if i < window:
            slopes.append(0.0)
            r2s.append(0.0)
            continue
        y = vals[i - window:i]
        x = np.arange(window)
        if np.std(y) == 0:
            slopes.append(0.0)
            r2s.append(0.0)
        else:
            slope, _ = np.polyfit(x, y, 1)
            corr = np.corrcoef(x, y)[0, 1]
            slopes.append(slope)
            r2s.append(corr ** 2 if not np.isnan(corr) else 0.0)
    return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)


def detectar_patrones_smc_historico(df, ventana_estructura=15):
    n = len(df)
    senales_alcistas = np.zeros(n, dtype=bool)
    senales_bajistas = np.zeros(n, dtype=bool)
    ultima_senal = None

    if n < ventana_estructura + 10:
        return ultima_senal, senales_alcistas, senales_bajistas

    vol_media_20 = df['volumen'].rolling(20).mean()

    for i in range(ventana_estructura + 5, n):
        penultima_idx = i - 1
        inicio_ventana = max(0, i - ventana_estructura)
        fin_ventana = max(0, i - 2)
        if fin_ventana <= inicio_ventana:
            continue

        max_estructura = df['maximo'].iloc[inicio_ventana:fin_ventana].max()
        min_estructura = df['minimo'].iloc[inicio_ventana:fin_ventana].min()

        cierre_penultima = df['cierre'].iloc[penultima_idx]
        vol_penultima = df['volumen'].iloc[penultima_idx]
        idx_vol_media = penultima_idx - 1
        if idx_vol_media < 0 or pd.isna(vol_media_20.iloc[idx_vol_media]):
            continue
        vol_media_penultima = vol_media_20.iloc[idx_vol_media]

        if cierre_penultima > max_estructura and vol_penultima > vol_media_penultima:
            senales_alcistas[i] = True
        if cierre_penultima < min_estructura and vol_penultima > vol_media_penultima:
            senales_bajistas[i] = True

    if senales_alcistas[-1]:
        sl = df['minimo'].iloc[-10:].min() * 0.995
        tp = df['cierre'].iloc[-1] + (df['cierre'].iloc[-1] - sl) * 2.0
        ultima_senal = {"tipo": "SMC_ALCISTA", "sl": sl, "tp": tp}
    elif senales_bajistas[-1]:
        sl = df['maximo'].iloc[-10:].max() * 1.005
        tp = df['cierre'].iloc[-1] - (sl - df['cierre'].iloc[-1]) * 2.0
        ultima_senal = {"tipo": "SMC_BAJISTA", "sl": sl, "tp": tp}

    return ultima_senal, senales_alcistas, senales_bajistas


def validar_senal_historica(df, condicion_activa, horizonte=HORIZONTE_VALIDACION):
    disparos = np.where(condicion_activa)[0]
    disparos = disparos[disparos < len(df) - horizonte]
    if len(disparos) < 5:
        return {'n_casos': len(disparos), 'win_rate': None, 'retorno_promedio': None, 'suficiente': False, 'mc_confianza': None}

    costo_total = (COMISION_EXCHANGE + SLIPPAGE) * 2
    retornos = np.array([((df.iloc[i + horizonte]['cierre'] / df.iloc[i]['cierre']) - 1 - costo_total) * 100 for i in disparos])

    np.random.seed(42)
    simulaciones_mc = [np.random.choice(retornos, size=len(retornos), replace=True).sum() for _ in range(1000)]
    prob_positiva_mc = (np.array(simulaciones_mc) > 0).mean() * 100
    return {
        'n_casos': len(disparos), 'win_rate': (retornos > 0).mean() * 100,
        'retorno_promedio': retornos.mean(), 'suficiente': len(disparos) >= 15,
        'mc_confianza': prob_positiva_mc
    }


def es_mercado_spot_valido(exchange, simbolo):
    mercado = exchange.markets.get(simbolo)
    return mercado and mercado.get('spot', False) is True and mercado.get('type', 'spot') == 'spot'


# ==========================================================================
# 3. MÓDULOS DE MEMORIA E IA AISLADA
# ==========================================================================
def registrar_prediccion(simbolo, precio, rsi, atr, adx, vol, hurst, slope, r2, crt, prob_subida):
    nueva_fila = pd.DataFrame([{
        'timestamp': datetime.now(timezone.utc).isoformat(), 'simbolo': simbolo, 'precio_entrada': precio,
        'RSI': rsi, 'ATR': atr, 'ADX': adx, 'volumen': vol,
        'Hurst': hurst, 'Pendiente_Reg': slope, 'R2_Tendencia': r2, 'CRT_Valida': crt,
        'prob_subida_predicha': prob_subida, 'precio_futuro': np.nan, 'exito_real': np.nan
    }])
    if not os.path.exists(ARCHIVO_MEMORIA):
        nueva_fila.to_csv(ARCHIVO_MEMORIA, index=False)
    else:
        nueva_fila.to_csv(ARCHIVO_MEMORIA, mode='a', header=False, index=False)


def auditar_memoria(exchange, horizonte_dias=HORIZONTE_VALIDACION):
    if not os.path.exists(ARCHIVO_MEMORIA):
        return
    try:
        if os.path.getsize(ARCHIVO_MEMORIA) == 0:
            return
        df_memoria = pd.read_csv(ARCHIVO_MEMORIA)
        if df_memoria.empty:
            return
    except Exception:
        return

    df_memoria['timestamp'] = pd.to_datetime(df_memoria['timestamp'])
    ahora = datetime.now(timezone.utc)
    pendientes = df_memoria[df_memoria['exito_real'].isna()]
    cambios = False
    for idx, fila in pendientes.iterrows():
        delta_dias = (ahora - fila['timestamp']).days
        if delta_dias >= horizonte_dias:
            try:
                fecha_objetivo = fila['timestamp'] + pd.Timedelta(days=horizonte_dias)
                since_ms = int(fecha_objetivo.timestamp() * 1000)
                velas = exchange.fetch_ohlcv(fila['simbolo'], timeframe='1d', since=since_ms, limit=2)
                if velas:
                    precio_futuro = velas[0][4]
                    df_memoria.at[idx, 'precio_futuro'] = precio_futuro
                    df_memoria.at[idx, 'exito_real'] = 1 if precio_futuro > fila['precio_entrada'] else 0
                    cambios = True
                time.sleep(0.2)
            except Exception:
                pass
    if cambios:
        df_memoria.to_csv(ARCHIVO_MEMORIA, index=False)


def calcular_riesgo_dinamico(riesgo_base=0.01):
    if not os.path.exists(ARCHIVO_MEMORIA):
        return riesgo_base, "SIN DATOS"
    try:
        if os.path.getsize(ARCHIVO_MEMORIA) == 0:
            return riesgo_base, "SIN DATOS"
        df_mem = pd.read_csv(ARCHIVO_MEMORIA)
        if df_mem.empty:
            return riesgo_base, "SIN DATOS"
    except Exception:
        return riesgo_base, "SIN DATOS"

    auditadas = df_mem.dropna(subset=['exito_real'])
    if len(auditadas) < MIN_CASOS_RIESGO_DINAMICO:
        return riesgo_base, f"RECOPILANDO ({len(auditadas)}/{MIN_CASOS_RIESGO_DINAMICO})"
    win_rate_reciente = auditadas.tail(MIN_CASOS_RIESGO_DINAMICO)['exito_real'].mean()
    if win_rate_reciente < 0.40:
        return riesgo_base * 0.5, f"PERDEDORA ({win_rate_reciente*100:.0f}%)"
    elif win_rate_reciente > 0.60:
        return riesgo_base * 1.5, f"GANADORA ({win_rate_reciente*100:.0f}%)"
    return riesgo_base, f"ESTABLE ({win_rate_reciente*100:.0f}%)"


def detectar_regimen_kmeans(df_btc):
    if len(df_btc) < 100:
        return "DESCONOCIDO"
    try:
        data = pd.DataFrame(index=df_btc.index)
        data['rendimiento'] = df_btc['cierre'].pct_change()
        data['volatilidad'] = df_btc['maximo'] - df_btc['minimo']
        data['tendencia'] = df_btc['cierre'] / df_btc['cierre'].rolling(50).mean() - 1
        data = data.dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        data['cluster'] = kmeans.fit_predict(X_scaled)
        medias = data.groupby('cluster')['tendencia'].mean().sort_values()
        mapa_regimenes = {medias.index[0]: 'BAJISTA', medias.index[1]: 'LATERAL', medias.index[2]: 'ALCISTA'}
        return mapa_regimenes[data['cluster'].iloc[-1]]
    except Exception:
        return "LATERAL"


def inferir_probabilidad_xgboost(df, simbolo_actual):
    try:
        df_ml = df.copy()
        df_ml['retorno_futuro'] = df_ml['cierre'].shift(-HORIZONTE_VALIDACION) / df_ml['cierre'] - 1
        df_ml['exito'] = (df_ml['retorno_futuro'] > 0).astype(int)

        features = ['RSI', 'ATR', 'ADX', 'volumen', 'Hurst', 'Pendiente_Reg', 'R2_Tendencia', 'CRT_Valida']
        df_ml = df_ml.dropna(subset=features)

        X = df_ml[:-HORIZONTE_VALIDACION][features]
        y = df_ml[:-HORIZONTE_VALIDACION]['exito']

        if os.path.exists(ARCHIVO_MEMORIA):
            try:
                if os.path.getsize(ARCHIVO_MEMORIA) > 0:
                    df_mem = pd.read_csv(ARCHIVO_MEMORIA)
                    df_mem = df_mem[(df_mem['simbolo'] == simbolo_actual) & (df_mem['exito_real'].notna())]
                    if not df_mem.empty and all(f in df_mem.columns for f in features):
                        X_memoria = df_mem[features]
                        y_memoria = df_mem['exito_real']
                        X = pd.concat([X, X_memoria], ignore_index=True)
                        y = pd.concat([y, y_memoria], ignore_index=True)
            except Exception:
                pass

        if len(X) < 40 or y.nunique() < 2:
            return None, None

        xgb_model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X, y)

        X_actual = df_ml.iloc[-1:][features]
        probabilidades = xgb_model.predict_proba(X_actual)[0]
        return probabilidades[1] * 100, probabilidades[0] * 100
    except Exception:
        return None, None


# ==========================================================================
# 4. TELEGRAM (TEXTO PLANO Y FRAGMENTADO PARA EVITAR ERRORES DE SINTAXIS)
# ==========================================================================
def enviar_alerta_telegram(mensaje):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ ERROR: TELEGRAM_TOKEN o TELEGRAM_CHAT_ID no están definidos en el entorno.")
        return
    
    # Limpiamos asteriscos o acentos graves para prevenir cualquier error de formato en Telegram
    mensaje_plano = mensaje.replace('*', '').replace('`', '')
    
    max_length = 4000
    mensajes = [mensaje_plano[i:i+max_length] for i in range(0, len(mensaje_plano), max_length)]
    
    for idx, msg_chunk in enumerate(mensajes):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            # Omitimos parse_mode para enviar texto plano seguro
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg_chunk}
            response = requests.post(url, json=payload, timeout=15)
            
            print(f"📡 Código de respuesta de Telegram (Parte {idx+1}): {response.status_code}")
            if response.status_code != 200:
                print(f"📦 Error de Telegram: {response.text}")
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ EXCEPCIÓN al conectar con Telegram (Parte {idx+1}): {e}")


def cargar_estado():
    if os.path.exists(ESTADO_PATH):
        try:
            with open(ESTADO_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def guardar_estado(estado):
    with open(ESTADO_PATH, 'w') as f:
        json.dump(estado, f, indent=2)


def formatear_bloque_telegram(alertas):
    txt = ""
    for o in alertas:
        txt += f"• {o['simbolo']} | ${o['precio']:,.4f}\n  {o['etiqueta']}\n"
        txt += f"  Trailing Stop: ${o['trailing_stop']:,.4f}\n"
        if o.get('prob_subida') is not None:
            txt += f"  Probabilidad IA: 📈 {o['prob_subida']:.1f}% | 📉 {o['prob_bajada']:.1f}%\n"
    return txt


# ==========================================================================
# 5. ANÁLISIS DE ACTIVOS LARGO PLAZO
# ==========================================================================
def analizar_activo_largo_plazo(exchange, simbolo, temporalidad, descuento_pct, rsi_compra, rsi_venta, sobreprecio_pct, riesgo_aplicado, tipo_estrategia="REVERSION"):
    velas = descargar_velas_cerradas(exchange, simbolo, temporalidad, VELAS_ANALISIS)
    if not velas or len(velas) < 60:
        return None

    df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
    df['RSI'] = calcular_rsi(df['cierre'], 14)
    df['VWMA_30'] = calcular_vwma(df, 30)
    df['ATR'] = calcular_atr(df, 14)
    df['ADX'] = calcular_adx(df, 14)

    df['Hurst'] = calcular_hurst_vectorizado(df['cierre'])
    slopes, r2s = calcular_regresion_rolling(df['cierre'])
    df['Pendiente_Reg'] = slopes
    df['R2_Tendencia'] = r2s

    rango_velas = df['maximo'] - df['minimo']
    df['CRT'] = np.where(rango_velas == 0, 0, (df['cierre'] - df['apertura']).abs() / rango_velas)
    df['CRT_Valida'] = (df['CRT'] >= 0.70).astype(int)

    df = df.dropna().reset_index(drop=True)
    if len(df) == 0:
        return None

    ultima = df.iloc[-1]
    precio, rsi, vwma_30, atr = ultima['cierre'], ultima['RSI'], ultima['VWMA_30'], ultima['ATR']
    trailing_stop = precio - (atr * 2.5)
    prefijo = "*(Alta Volatilidad):* " if tipo_estrategia == "MOMENTUM" else ""

    if tipo_estrategia == "REVERSION":
        condicion_descuento = (df['cierre'] <= df['VWMA_30'] * (1 - descuento_pct)) | (df['RSI'] < rsi_compra)
        condicion_sobrecompra = df['RSI'] >= rsi_venta
        if precio <= (vwma_30 * (1 - descuento_pct)) or rsi < rsi_compra:
            accion, etiqueta, validacion = "COMPRAR", f"{prefijo}🟢 COMPRAR: Activo en descuento bajo VWMA.", validar_senal_historica(df, condicion_descuento.values)
        elif rsi >= rsi_venta:
            accion, etiqueta, validacion = "EVALUAR_VENTA", f"{prefijo}🟡 TOMAR BENEFICIOS: Euforia extrema.", validar_senal_historica(df, condicion_sobrecompra.values)
        elif precio > (vwma_30 * (1 + sobreprecio_pct)):
            accion, etiqueta, validacion = "ESPERAR", f"{prefijo}🔴 ESPERAR: Precio inflado sobre volumen.", None
        else:
            accion, etiqueta, validacion = "NEUTRO", f"{prefijo}⚪ ZONA NEUTRA: Mercado estable.", None

    elif tipo_estrategia == "MOMENTUM":
        condicion_momentum = (df['cierre'] > df['VWMA_30']) & (df['RSI'] > 55) & (df['RSI'] < 75)
        condicion_caida = (df['RSI'] >= 80) | (df['cierre'] < df['VWMA_30'] * 0.95)
        if precio > vwma_30 and 55 < rsi < 75:
            accion, etiqueta, validacion = "COMPRAR", f"🔥 COMPRAR (MOMENTUM): Tendencia fuerte sobre VWMA.", validar_senal_historica(df, condicion_momentum.values)
        elif rsi >= 80 or precio < (vwma_30 * 0.95):
            accion, etiqueta, validacion = "EVALUAR_VENTA", f"🟡 CORTAR / BENEFICIOS: Tendencia agotada.", validar_senal_historica(df, condicion_caida.values)
        else:
            accion, etiqueta, validacion = "NEUTRO", f"⚪ ZONA NEUTRA: Sin fuerza direccional.", None

    denominador_atr = max(atr * 2.5, 0.0001)
    sugerencia_tamano = CAPITAL_REFERENCIA_USD * riesgo_aplicado / denominador_atr
    prob_subida, prob_bajada = inferir_probabilidad_xgboost(df, simbolo)

    return {
        'simbolo': simbolo, 'precio': precio, 'rsi': rsi, 'vwma_30': vwma_30,
        'accion': accion, 'etiqueta': etiqueta, 'validacion': validacion,
        'sugerencia_tamano': sugerencia_tamano, 'atr': atr, 'trailing_stop': trailing_stop,
        'prob_subida': prob_subida, 'prob_bajada': prob_bajada
    }


# ==========================================================================
# 6. BOT MAESTRO
# ==========================================================================
def ejecutar_bot_maestro():
    exchange = ccxt.kucoin({'enableRateLimit': True, 'timeout': 30000})
    exchange.load_markets()

    auditar_memoria(exchange)
    riesgo_dinamico, estado_racha = calcular_riesgo_dinamico(RIESGO_POR_TRADE)

    estado_anterior, estado_nuevo = cargar_estado(), {}
    alertas_nucleo, alertas_crecimiento, alertas_seguimiento, alertas_alto_riesgo, alertas_satelite = [], [], [], [], []

    velas_btc = descargar_velas_cerradas(exchange, 'BTC/USDT', TEMPORALIDAD_ESCUDO, VELAS_ESCUDO_BTC)
    velas_eth = descargar_velas_cerradas(exchange, 'ETH/USDT', TEMPORALIDAD_ESCUDO, 60)

    regimen_macro = "DESCONOCIDO"
    altseason_estado = "DESCONOCIDO"

    if velas_btc:
        df_btc = pd.DataFrame(velas_btc, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
        regimen_macro = detectar_regimen_kmeans(df_btc)
        if velas_eth:
            df_eth = pd.DataFrame(velas_eth, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
            if len(df_btc) > 30 and len(df_eth) > 30:
                rend_btc = (df_btc['cierre'].iloc[-1] / df_btc['cierre'].iloc[-30]) - 1
                rend_eth = (df_eth['cierre'].iloc[-1] / df_eth['cierre'].iloc[-30]) - 1
                altseason_estado = "🟢 ON (Rotación a Alts)" if rend_eth > rend_btc else "🔴 OFF (Dominancia BTC)"

    for s in NUCLEO_CONSERVADOR:
        if r := analizar_activo_largo_plazo(exchange, s, TEMPORALIDAD_NUCLEO, 0.04, 40, 75, 0.05, riesgo_dinamico, "REVERSION"):
            alertas_nucleo.append(r)
            estado_nuevo[s] = r['accion']
        time.sleep(0.3)

    for s in NIVEL_CRECIMIENTO:
        if r := analizar_activo_largo_plazo(exchange, s, TEMPORALIDAD_CRECIMIENTO, 0.07, 35, 80, 0.08, riesgo_dinamico, "REVERSION"):
            alertas_crecimiento.append(r)
            estado_nuevo[s] = r['accion']
        time.sleep(0.3)

    for s in SEGUIMIENTO_ESTRATEGICO:
        if r := analizar_activo_largo_plazo(exchange, s, TEMPORALIDAD_SEGUIMIENTO, 0.06, 35, 78, 0.07, riesgo_dinamico, "REVERSION"):
            alertas_seguimiento.append(r)
            estado_nuevo[s] = r['accion']
        time.sleep(0.3)

    for s in ALTO_RIESGO:
        if r := analizar_activo_largo_plazo(exchange, s, TEMPORALIDAD_ALTO_RIESGO, 0.08, 30, 80, 0.10, riesgo_dinamico, "MOMENTUM"):
            alertas_alto_riesgo.append(r)
            estado_nuevo[s] = r['accion']
        time.sleep(0.3)

    try:
        tickers = exchange.fetch_tickers()
        excluidos = set(NUCLEO_CONSERVADOR) | set(NIVEL_CRECIMIENTO) | set(SEGUIMIENTO_ESTRATEGICO) | set(ALTO_RIESGO)
        candidatas = sorted([s for s, t in tickers.items() if '/USDT' in s and s not in excluidos and es_mercado_spot_valido(exchange, s) and t.get('quoteVolume', 0) > 1500000], key=lambda s: tickers[s].get('quoteVolume', 0), reverse=True)[:15]

        historico_candidatas_aceptadas = {}

        for s in candidatas:
            velas = descargar_velas_cerradas(exchange, s, TEMPORALIDAD_SATELITE, VELAS_ANALISIS)
            if not velas or len(velas) < 60:
                continue

            df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
            df['RSI'], _, df['BB_Lower'] = calcular_rsi(df['cierre'], 14), None, calcular_bollinger_bands(df['cierre'])[2]
            df['VWMA_20'], df['ATR'], df['ADX'] = calcular_vwma(df, 20), calcular_atr(df, 14), calcular_adx(df, 14)
            df['Max_20'] = df['cierre'].shift(1).rolling(20).max()

            df['Hurst'] = calcular_hurst_vectorizado(df['cierre'])
            slopes, r2s = calcular_regresion_rolling(df['cierre'])
            df['Pendiente_Reg'] = slopes
            df['R2_Tendencia'] = r2s

            rango_velas = df['maximo'] - df['minimo']
            df['CRT'] = np.where(rango_velas == 0, 0, (df['cierre'] - df['apertura']).abs() / rango_velas)
            df['CRT_Valida'] = (df['CRT'] >= 0.70).astype(int)

            df = df.dropna().reset_index(drop=True)
            if len(df) == 0:
                continue
            ultima = df.iloc[-1]

            patron_smc, hist_alc_smc, hist_baj_smc = detectar_patrones_smc_historico(df)

            if regimen_macro == "BAJISTA":
                cuantitativo_ok = False
                breakout_ok = False
                patron_smc = None
            elif regimen_macro == "LATERAL":
                cuantitativo_ok = (ultima['RSI'] <= SATELITE_RSI_ENTRADA and ultima['cierre'] <= ultima['BB_Lower'] * 1.01 and ultima['cierre'] < ultima['VWMA_20'])
                breakout_ok = False
            else:
                cuantitativo_ok = (ultima['RSI'] <= SATELITE_RSI_ENTRADA and ultima['cierre'] <= ultima['BB_Lower'] * 1.01 and ultima['ADX'] < SATELITE_ADX_MAX)
                breakout_ok = (ultima['cierre'] > ultima['Max_20']) and (ultima['cierre'] > ultima['VWMA_20']) and (ultima['ADX'] > 25) and ultima['CRT_Valida'] == 1

            if cuantitativo_ok or breakout_ok or patron_smc:
                correlacion_peligrosa = False
                retornos_actuales = df['cierre'].tail(50).pct_change().dropna().values
                for s_aceptado, retornos_aceptados in historico_candidatas_aceptadas.items():
                    if len(retornos_actuales) == len(retornos_aceptados):
                        corr = np.corrcoef(retornos_actuales, retornos_aceptados)[0, 1]
                        if not np.isnan(corr) and corr > 0.85:
                            correlacion_peligrosa = True
                            break
                if correlacion_peligrosa:
                    continue

                historico_candidatas_aceptadas[s] = retornos_actuales
                prob_subida, prob_bajada = inferir_probabilidad_xgboost(df, s)

                if cuantitativo_ok:
                    tipo_al = 'CUANTITATIVO_REVERSION'
                    sl, tp = ultima['cierre'] - (SATELITE_ATR_SL_MULT * ultima['ATR']), ultima['cierre'] + (SATELITE_ATR_TP_MULT * ultima['ATR'])
                    val = validar_senal_historica(df, (df['RSI'] <= SATELITE_RSI_ENTRADA) & (df['cierre'] <= df['BB_Lower'] * 1.01))
                elif breakout_ok:
                    tipo_al = 'BREAKOUT_MOMENTUM_CRT'
                    sl, tp = ultima['cierre'] - (1.5 * ultima['ATR']), ultima['cierre'] + (3.0 * ultima['ATR'])
                    val = validar_senal_historica(df, (df['cierre'] > df['cierre'].shift(1).rolling(20).max()) & (df['cierre'] > ultima['VWMA_20']))
                else:
                    tipo_al = patron_smc['tipo']
                    sl, tp = patron_smc['sl'], patron_smc['tp']
                    condicion_smc = hist_alc_smc if patron_smc['tipo'] == "SMC_ALCISTA" else hist_baj_smc
                    val = validar_senal_historica(df, condicion_smc)

                if prob_subida is not None:
                    registrar_prediccion(s, ultima['cierre'], ultima['RSI'], ultima['ATR'], ultima['ADX'], ultima['volumen'], ultima['Hurst'], ultima['Pendiente_Reg'], ultima['R2_Tendencia'], ultima['CRT_Valida'], prob_subida)
                alertas_satelite.append({'simbolo': s, 'precio': ultima['cierre'], 'tipo_alerta': tipo_al, 'tp': tp, 'sl': sl, 'sugerencia_tamano': CAPITAL_REFERENCIA_USD * riesgo_dinamico / max(abs(ultima['cierre'] - sl), 0.0001), 'validacion': val, 'prob_subida': prob_subida, 'prob_bajada': prob_bajada})
                estado_nuevo[s] = tipo_al

            time.sleep(0.3)

    except Exception as e:
        print(f"⚠️ Error procesando el bloque satélite: {e}")

    # ==========================================================================
    # ENVÍO INCONDICIONAL A TELEGRAM (TEXTO PLANO / SEGURO)
    # ==========================================================================
    todas_las_alertas_largo_plazo = alertas_nucleo + alertas_crecimiento + alertas_seguimiento + alertas_alto_riesgo

    msj = f"🚨 REPORTE IA (INSTITUCIONAL) — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
    msj += f"🌐 Régimen Macro: {regimen_macro}\n"
    msj += f"🔄 Flujo de Capital: {altseason_estado}\n"
    msj += f"🧠 Racha IA: {estado_racha} | Riesgo actual: {riesgo_dinamico*100:.2f}%\n\n"

    if todas_las_alertas_largo_plazo:
        msj += "📊 ESTADO GENERAL DEL PORTAFOLIO\n"
        for o in todas_las_alertas_largo_plazo:
            msj += f"• {o['simbolo']} | ${o['precio']:,.4f}\n  {o['etiqueta']}\n"
            msj += f"  🛡️ Trailing Stop: ${o['trailing_stop']:,.4f}\n"
            if o.get('prob_subida') is not None:
                msj += f"  🤖 Probabilidad IA: 📈 {o['prob_subida']:.1f}% | 📉 {o['prob_bajada']:.1f}%\n"
        msj += "\n"

    if alertas_satelite:
        msj += f"🎯 RADAR DE PRECISIÓN ({TEMPORALIDAD_SATELITE})\n"
        for o in alertas_satelite:
            emoji = '🟢' if 'ALCISTA' in o['tipo_alerta'] or 'REVERSION' in o['tipo_alerta'] or 'BREAKOUT' in o['tipo_alerta'] else '🔴'
            msj += f"{emoji} {o['tipo_alerta'].replace('_', ' ')} | {o['simbolo']}\n  Entrada: ${o['precio']:,.4f}\n  🎯 TP: ${o['tp']:,.4f} | 🛑 SL: ${o['sl']:,.4f}\n"
            if o.get('prob_subida') is not None:
                msj += f"  🤖 Probabilidad IA: 📈 {o['prob_subida']:.1f}% | 📉 {o['prob_bajada']:.1f}%\n"
            if o['validacion'] and o['validacion']['suficiente']:
                msj += f"  📊 Histórico (con comisiones): Win Rate {o['validacion']['win_rate']:.0f}%\n"
            elif o['validacion'] and o['validacion']['n_casos'] > 0:
                msj += f"  ⚠️ Solo {o['validacion']['n_casos']} casos históricos — poco confiable\n"
        msj += "\n"
    else:
        msj += f"🎯 RADAR DE PRECISIÓN ({TEMPORALIDAD_SATELITE})\n  ⚪ Sin señales activas en este ciclo.\n\n"

    msj += "_⚠️ Herramienta cuantitativa. No constituye asesoría financiera personalizada._"

    enviar_alerta_telegram(msj)
    guardar_estado(estado_nuevo)


if __name__ == '__main__':
    ejecutar_bot_maestro()
