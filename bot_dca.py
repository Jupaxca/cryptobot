import ccxt
import pandas as pd
import numpy as np
import os
import json
import requests
from datetime import datetime, timezone
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ==========================================================================
# 1. CONFIGURACIÓN GENERAL DEL PORTAFOLIO
# ==========================================================================
NUCLEO_CONSERVADOR = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LINK/USDT', 'BNB/USDT']
TEMPORALIDAD_NUCLEO = '1d'

NIVEL_CRECIMIENTO = ['NEAR/USDT', 'ONDO/USDT', 'TAO/USDT', 'AVAX/USDT']
TEMPORALIDAD_CRECIMIENTO = '1d'

SEGUIMIENTO_ESTRATEGICO = ['XRP/USDT', 'HYPE/USDT', 'ADA/USDT', 'POL/USDT', 'SUI/USDT', 'PUMP/USDT', 'UNI/USDT', 'ZEC/USDT']
TEMPORALIDAD_SEGUIMIENTO = '1d'

ALTO_RIESGO= ['THETA/USDT', 'ASTER/USDT']
TEMPORALIDAD_SEGUIMIENTO = '1d'

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
ARCHIVO_MEMORIA = 'memoria_predicciones.csv' # NUEVO: Archivo de memoria IA
SOLO_ALERTAR_CAMBIOS = False

CAPITAL_REFERENCIA_USD = 90  
RIESGO_POR_TRADE = 0.01      
HORIZONTE_VALIDACION = 14

# ==========================================================================
# 2. DESCARGA Y PROCESAMIENTO TÉCNICO
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
    senales_alcistas = np.zeros(len(df), dtype=bool)
    senales_bajistas = np.zeros(len(df), dtype=bool)
    ultima_senal = None
    
    if len(df) < 20: return ultima_senal, senales_alcistas, senales_bajistas

    for i in range(15, len(df)):
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
        return {'n_casos': len(disparos), 'win_rate': None, 'retorno_promedio': None, 'suficiente': False, 'mc_confianza': None}
        
    retornos = np.array([(df.iloc[i + horizonte]['cierre'] / df.iloc[i]['cierre'] - 1) * 100 for i in disparos])
    
    # NUEVO: Simulación de Monte Carlo (1000 iteraciones)
    np.random.seed(42)
    simulaciones_mc = [np.random.choice(retornos, size=len(retornos), replace=True).sum() for _ in range(1000)]
    prob_positiva_mc = (np.array(simulaciones_mc) > 0).mean() * 100
    
    return {
        'n_casos': len(disparos), 
        'win_rate': (retornos > 0).mean() * 100,
        'retorno_promedio': retornos.mean(), 
        'suficiente': len(disparos) >= 15,
        'mc_confianza': prob_positiva_mc
    }

def es_mercado_spot_valido(exchange, simbolo):
    mercado = exchange.markets.get(simbolo)
    return mercado and mercado.get('spot', False) is True and mercado.get('type', 'spot') == 'spot'

# ==========================================================================
# 3. MÓDULOS DE MEMORIA E IA AVANZADA
# ==========================================================================

# --- NUEVO: SISTEMA DE MEMORIA (FEEDBACK LOOP) ---
def registrar_prediccion(simbolo, precio_entrada, rsi, atr, adx, volumen, prob_subida):
    nueva_fila = pd.DataFrame([{
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'simbolo': simbolo,
        'precio_entrada': precio_entrada,
        'RSI': rsi, 'ATR': atr, 'ADX': adx, 'volumen': volumen,
        'prob_subida_predicha': prob_subida,
        'precio_futuro': np.nan,  
        'exito_real': np.nan      
    }])
    if not os.path.exists(ARCHIVO_MEMORIA):
        nueva_fila.to_csv(ARCHIVO_MEMORIA, index=False)
    else:
        nueva_fila.to_csv(ARCHIVO_MEMORIA, mode='a', header=False, index=False)

def auditar_memoria(exchange, horizonte_dias=HORIZONTE_VALIDACION):
    if not os.path.exists(ARCHIVO_MEMORIA): return
    df_memoria = pd.read_csv(ARCHIVO_MEMORIA)
    df_memoria['timestamp'] = pd.to_datetime(df_memoria['timestamp'])
    ahora = datetime.now(timezone.utc)
    pendientes = df_memoria[df_memoria['exito_real'].isna()]
    cambios = False
    
    for idx, fila in pendientes.iterrows():
        if (ahora - fila['timestamp']).days >= horizonte_dias:
            try:
                velas = exchange.fetch_ohlcv(fila['simbolo'], timeframe='1d', limit=2)
                if velas:
                    precio_actual = velas[-1][4]
                    df_memoria.at[idx, 'precio_futuro'] = precio_actual
                    df_memoria.at[idx, 'exito_real'] = 1 if precio_actual > fila['precio_entrada'] else 0
                    cambios = True
            except Exception as e:
                print(f"⚠️ Error auditando {fila['simbolo']}: {e}")
                
    if cambios:
        df_memoria.to_csv(ARCHIVO_MEMORIA, index=False)
        print("🧠 Auditoría completada: El bot ha evaluado sus predicciones pasadas.")

def calcular_riesgo_dinamico(riesgo_base=0.01):
    if not os.path.exists(ARCHIVO_MEMORIA):
        return riesgo_base, "SIN DATOS" 
        
    df_mem = pd.read_csv(ARCHIVO_MEMORIA)
    auditadas = df_mem.dropna(subset=['exito_real'])
    
    if len(auditadas) < 10:
        return riesgo_base, "RECOPILANDO" 
        
    win_rate_reciente = auditadas.tail(10)['exito_real'].mean() 
    
    if win_rate_reciente < 0.40:
        return riesgo_base * 0.5, f"PERDEDORA ({win_rate_reciente*100:.0f}%)"
    elif win_rate_reciente > 0.60:
        return riesgo_base * 1.5, f"GANADORA ({win_rate_reciente*100:.0f}%)"
        
    return riesgo_base, f"ESTABLE ({win_rate_reciente*100:.0f}%)"

def detectar_regimen_kmeans(df_btc):
    if len(df_btc) < 50: return "DESCONOCIDO"
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
    except Exception as e: 
        print(f"⚠️ Error en K-Means: {e}")
        return "LATERAL"

def inferir_probabilidad_xgboost(df):
    try:
        df_ml = df.copy()
        df_ml['retorno_futuro'] = df_ml['cierre'].shift(-HORIZONTE_VALIDACION) / df_ml['cierre'] - 1
        df_ml['exito'] = (df_ml['retorno_futuro'] > 0).astype(int)
        
        features = ['RSI', 'ATR', 'ADX', 'volumen']
        df_ml = df_ml.dropna(subset=features)
        
        X = df_ml[:-HORIZONTE_VALIDACION][features]
        y = df_ml[:-HORIZONTE_VALIDACION]['exito']
        
        # --- NUEVO: Inyectar Memoria ---
        if os.path.exists(ARCHIVO_MEMORIA):
            df_mem = pd.read_csv(ARCHIVO_MEMORIA).dropna(subset=['exito_real'])
            if not df_mem.empty:
                X_memoria = df_mem[features]
                y_memoria = df_mem['exito_real']
                X = pd.concat([X, X_memoria, X_memoria], ignore_index=True)
                y = pd.concat([y, y_memoria, y_memoria], ignore_index=True)
        # -------------------------------
        
        if len(X) < 50 or y.nunique() < 2: return None, None
        
        xgb_model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', use_label_encoder=False
        )
        xgb_model.fit(X, y)
        
        X_actual = df_ml.iloc[-1:][features]
        probabilidades = xgb_model.predict_proba(X_actual)[0]
        return probabilidades[1] * 100, probabilidades[0] * 100
    except Exception as e: 
        print(f"⚠️ Error en XGBoost: {e}")
        return None, None

# ==========================================================================
# 4. TELEGRAM Y ESTADO
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
# 5. ANÁLISIS DE ACTIVOS
# ==========================================================================
def analizar_activo_largo_plazo(exchange, simbolo, temporalidad, descuento_pct, rsi_compra, rsi_venta, sobreprecio_pct, riesgo_aplicado, es_crecimiento=False):
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

    denominador_atr = max(atr * 2.5, 0.0001)
    sugerencia_tamano = CAPITAL_REFERENCIA_USD * riesgo_aplicado / denominador_atr

    return {'simbolo': simbolo, 'precio': precio, 'rsi': rsi, 'media_30': media_30, 'accion': accion, 'etiqueta': etiqueta, 'validacion': validacion, 'sugerencia_tamano': sugerencia_tamano, 'atr': atr}

## ==========================================================================
# 6. BOT MAESTRO
# ==========================================================================
def ejecutar_bot_maestro():
    exchange = ccxt.kucoin({'enableRateLimit': True, 'timeout': 30000})
    exchange.load_markets()
    
    # 1. Auditoría automática al arrancar
    auditar_memoria(exchange)
    # 2. Calcular hiperparámetro de riesgo dinámico
    riesgo_dinamico, estado_racha = calcular_riesgo_dinamico(RIESGO_POR_TRADE)
    print(f"⚖️ Riesgo dinámico ajustado a: {riesgo_dinamico*100:.2f}% | Racha: {estado_racha}")
    
    estado_anterior, estado_nuevo = cargar_estado(), {}
    alertas_nucleo, alertas_crecimiento, alertas_seguimiento, alertas_satelite = [], [], [], []

    velas_btc = descargar_velas_cerradas(exchange, 'BTC/USD', TEMPORALIDAD_ESCUDO, VELAS_ESCUDO_BTC)
    regimen_macro = "DESCONOCIDO"
    if velas_btc:
        df_btc = pd.DataFrame(velas_btc, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
        regimen_macro = detectar_regimen_kmeans(df_btc)

    print(f"🧠 Régimen Macro detectado por K-Means: {regimen_macro}")

    # --- Bloques Base ---
    for s in NUCLEO_CONSERVADOR:
        if r := analizar_activo_largo_plazo(exchange, s, TEMPORALIDAD_NUCLEO, 0.04, 40, 75, 0.05, riesgo_dinamico, False):
            alertas_nucleo.append(r)
            estado_nuevo[s] = r['accion']

    for s in NIVEL_CRECIMIENTO:
        if r := analizar_activo_largo_plazo(exchange, s, TEMPORALIDAD_CRECIMIENTO, 0.07, 35, 80, 0.08, riesgo_dinamico, True):
            alertas_crecimiento.append(r)
            estado_nuevo[s] = r['accion']

    for s in SEGUIMIENTO_ESTRATEGICO:
        if r := analizar_activo_largo_plazo(exchange, s, TEMPORALIDAD_SEGUIMIENTO, 0.06, 35, 78, 0.07, riesgo_dinamico, True):
            alertas_seguimiento.append(r)
            estado_nuevo[s] = r['accion']

    # --- Satélite (Híbrido + IA XGBoost) ---
    try:
        tickers = exchange.fetch_tickers()
        excluidos = set(NUCLEO_CONSERVADOR) | set(NIVEL_CRECIMIENTO) | set(SEGUIMIENTO_ESTRATEGICO)
        candidatas = sorted([s for s, t in tickers.items() if '/USD' in s and s not in excluidos and 'USDT' not in s and es_mercado_spot_valido(exchange, s) and t.get('quoteVolume', 0) > 1000000], key=lambda s: tickers[s].get('quoteVolume', 0), reverse=True)[:10]

        for s in candidatas:
            velas = descargar_velas_cerradas(exchange, s, TEMPORALIDAD_SATELITE, VELAS_ANALISIS)
            if not velas or len(velas) < 60: continue
            
            df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
            df['RSI'], _, df['BB_Lower'] = calcular_rsi(df['cierre'], 14), None, calcular_bollinger_bands(df['cierre'])[2]
            df['Vol_Medio'], df['ATR'], df['ADX'], df['EMA_TENDENCIA'] = df['volumen'].rolling(20).mean(), calcular_atr(df, 14), calcular_adx(df, 14), df['cierre'].ewm(span=200, adjust=False).mean()
            df['Max_20'] = df['cierre'].shift(1).rolling(20).max()
            df = df.dropna().reset_index(drop=True)
            ultima = df.iloc[-1]
            
            patron_smc, hist_alc, hist_baj = detectar_patrones_smc_hist(df)
            
            if regimen_macro == "BAJISTA":
                cuantitativo_ok = False
                breakout_ok = False
            elif regimen_macro == "LATERAL":
                cuantitativo_ok = (ultima['RSI'] <= SATELITE_RSI_ENTRADA and ultima['cierre'] <= ultima['BB_Lower'] * 1.01 and ultima['volumen'] >= ultima['Vol_Medio'] * 0.7)
                breakout_ok = False 
            else: # ALCISTA
                cuantitativo_ok = (ultima['RSI'] <= SATELITE_RSI_ENTRADA and ultima['cierre'] <= ultima['BB_Lower'] * 1.01 and ultima['ADX'] < SATELITE_ADX_MAX)
                breakout_ok = (ultima['cierre'] > ultima['Max_20']) and (ultima['volumen'] >= ultima['Vol_Medio'] * 1.5) and (ultima['ADX'] > 25)

            if cuantitativo_ok:
                sl, tp = ultima['cierre'] - (SATELITE_ATR_SL_MULT * ultima['ATR']), ultima['cierre'] + (SATELITE_ATR_TP_MULT * ultima['ATR'])
                val = validar_senal_historica(df, (df['RSI'] <= SATELITE_RSI_ENTRADA) & (df['cierre'] <= df['BB_Lower'] * 1.01))
                prob_subida, prob_bajada = inferir_probabilidad_xgboost(df)
                
                if prob_subida is not None: registrar_prediccion(s, ultima['cierre'], ultima['RSI'], ultima['ATR'], ultima['ADX'], ultima['volumen'], prob_subida)
                alertas_satelite.append({'simbolo': s, 'precio': ultima['cierre'], 'tipo_alerta': 'CUANTITATIVO_REVERSION', 'tp': tp, 'sl': sl, 'sugerencia_tamano': CAPITAL_REFERENCIA_USD * riesgo_dinamico / max(abs(ultima['cierre'] - sl), 0.0001), 'validacion': val, 'prob_subida': prob_subida, 'prob_bajada': prob_bajada})
                estado_nuevo[s] = 'CUANTITATIVO_REVERSION'
                
            elif breakout_ok:
                sl, tp = ultima['cierre'] - (1.5 * ultima['ATR']), ultima['cierre'] + (3.0 * ultima['ATR'])
                val = validar_senal_historica(df, (df['cierre'] > df['cierre'].shift(1).rolling(20).max()) & (df['volumen'] >= ultima['Vol_Medio'] * 1.5))
                prob_subida, prob_bajada = inferir_probabilidad_xgboost(df)

                if prob_subida is not None: registrar_prediccion(s, ultima['cierre'], ultima['RSI'], ultima['ATR'], ultima['ADX'], ultima['volumen'], prob_subida)
                alertas_satelite.append({'simbolo': s, 'precio': ultima['cierre'], 'tipo_alerta': 'BREAKOUT_MOMENTUM', 'tp': tp, 'sl': sl, 'sugerencia_tamano': CAPITAL_REFERENCIA_USD * riesgo_dinamico / max(abs(ultima['cierre'] - sl), 0.0001), 'validacion': val, 'prob_subida': prob_subida, 'prob_bajada': prob_bajada})
                estado_nuevo[s] = 'BREAKOUT_MOMENTUM'
                
            elif patron_smc:
                val_smc = validar_senal_historica(df, hist_alc if patron_smc['tipo'] == "SMC_ALCISTA" else hist_baj)
                prob_subida, prob_bajada = inferir_probabilidad_xgboost(df)
                
                if prob_subida is not None: registrar_prediccion(s, ultima['cierre'], ultima['RSI'], ultima['ATR'], ultima['ADX'], ultima['volumen'], prob_subida)
                alertas_satelite.append({'simbolo': s, 'precio': ultima['cierre'], 'tipo_alerta': patron_smc['tipo'], 'tp': patron_smc['tp'], 'sl': patron_smc['sl'], 'sugerencia_tamano': CAPITAL_REFERENCIA_USD * riesgo_dinamico / max(abs(ultima['cierre'] - patron_smc['sl']), 0.0001), 'validacion': val_smc, 'prob_subida': prob_subida, 'prob_bajada': prob_bajada})
                estado_nuevo[s] = patron_smc['tipo']
    except Exception as e:
        print(f"⚠️ Error procesando el bloque satélite: {e}")

    hubo_cambio = lambda s, a: estado_anterior.get(s) != a
    filtrar = lambda l, c: [a for a in l if not SOLO_ALERTAR_CAMBIOS or hubo_cambio(a['simbolo'], a.get('accion', c))]
    
    n_env = filtrar(alertas_nucleo, None)
    c_env = filtrar(alertas_crecimiento, None)
    seg_env = filtrar(alertas_seguimiento, None)
    s_env = [a for a in alertas_satelite if not SOLO_ALERTAR_CAMBIOS or hubo_cambio(a['simbolo'], a['tipo_alerta'])]

    if n_env or c_env or seg_env or s_env:
        msj = f"🚨 *REPORTE IA (MULTI-BLOQUE)* — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        msj += f"🌐 *Régimen Macro:* `{regimen_macro}`\n"
        msj += f"🧠 *Racha IA:* `{estado_racha}` | Riesgo actual: `{riesgo_dinamico*100:.2f}%`\n\n"
        
        if n_env:
            msj += "🛡️ *NÚCLEO CONSERVADOR*\n" + "".join([f"• *{o['simbolo']}* | `${o['precio']:,.2f}`\n  {o['etiqueta']}\n" for o in n_env]) + "\n"
        if c_env:
            msj += "🚀 *CRECIMIENTO PRINCIPAL*\n" + "".join([f"• *{o['simbolo']}* | `${o['precio']:,.2f}`\n  {o['etiqueta']}\n" for o in c_env]) + "\n"
        if seg_env:
            msj += "📊 *SEGUIMIENTO ESTRATÉGICO*\n" + "".join([f"• *{o['simbolo']}* | `${o['precio']:,.4f}`\n  {o['etiqueta']}\n" for o in seg_env]) + "\n"
        if s_env:
            msj += f"🎯 *RADAR DE ALTO RIESGO ({TEMPORALIDAD_SATELITE})*\n"
            for o in s_env:
                emoji = '🟢' if 'ALCISTA' in o['tipo_alerta'] or 'REVERSION' in o['tipo_alerta'] or 'BREAKOUT' in o['tipo_alerta'] else '🔴'
                msj += f"{emoji} *{o['tipo_alerta'].replace('_', ' ')}* | *{o['simbolo']}*\n  Entrada: `${o['precio']:,.4f}`\n  🎯 TP: `${o['tp']:,.4f}` | 🛑 SL: `${o['sl']:,.4f}`\n  📏 Tamaño sugerido: `{o['sugerencia_tamano']:,.2f}` UND\n"
                
                if o.get('prob_subida') is not None:
                    msj += f"  🤖 *Probabilidad IA:* 📈 Alza: `{o['prob_subida']:.1f}%` | 📉 Baja: `{o['prob_bajada']:.1f}%`\n"
                    
                if o['validacion'] and o['validacion']['suficiente']:
                    msj += f"  📊 Histórico: Win Rate {o['validacion']['win_rate']:.0f}%, Promedio {o['validacion']['retorno_promedio']:+.1f}%\n"
                    if o['validacion']['mc_confianza'] is not None:
                        msj += f"  🎲 *Monte Carlo Edge:* `{o['validacion']['mc_confianza']:.1f}%` prob. de ser rentable.\n"
                msj += "\n"
        msj += "_⚠️ Herramienta de apoyo analítico basada en IA. Verifique siempre antes de operar._"
        enviar_alerta_telegram(msj)
    
    guardar_estado(estado_nuevo)

if __name__ == '__main__':
    ejecutar_bot_maestro()
