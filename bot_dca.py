import ccxt
import pandas as pd
import numpy as np
import os
import json
import requests
from datetime import datetime, timezone

# ==========================================================================
# 1. CONFIGURACIÓN GENERAL DEL PORTAFOLIO
# ==========================================================================
# --- Núcleo Conservador: Base del portafolio (Baja volatilidad relativa) ---
NUCLEO_CONSERVADOR = ['BTC/USD', 'ETH/USD']
TEMPORALIDAD_NUCLEO = '1d'

# --- Nivel Crecimiento y Alto Potencial: Proyectos de alta utilidad pero mayor volatilidad ---
NIVEL_CRECIMIENTO = ['SOL/USD', 'LINK/USD', 'AVAX/USD']
TEMPORALIDAD_CRECIMIENTO = '1d'

# --- Satélite de Alto Riesgo (5%): Swing trading dinámico en 4h ---
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

CAPITAL_REFERENCIA = 370000
RIESGO_POR_TRADE = 0.01
HORIZONTE_VALIDACION = 14

# ==========================================================================
# 2. DESCARGA DE VELAS CERRADAS (Evita repintado)
# ==========================================================================
def descargar_velas_cerradas(exchange, simbolo, temporalidad, limit):
    try:
        tf_ms = exchange.parse_timeframe(temporalidad) * 1000
        velas = exchange.fetch_ohlcv(simbolo, timeframe=temporalidad, limit=limit + 1)
        if not velas:
            return velas
        ahora = exchange.milliseconds()
        ultima_vela_ts = velas[-1][0]
        if ultima_vela_ts + tf_ms > ahora:
            velas = velas[:-1]   # Descarta la vela en formación
        return velas[-limit:] if len(velas) > limit else velas
    except:
        return None

# ==========================================================================
# 3. INDICADORES TÉCNICOS Y SMC (SMART MONEY CONCEPTS)
# ==========================================================================
def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], 100)

def calcular_bollinger_bands(series, period=20, std_dev=2):
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return middle + (std * std_dev), middle, middle - (std * std_dev)

def calcular_atr(df, period=14):
    tr0 = df['maximo'] - df['minimo']
    tr1 = (df['maximo'] - df['cierre'].shift(1)).abs()
    tr2 = (df['minimo'] - df['cierre'].shift(1)).abs()
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calcular_adx(df, period=14):
    up_move = df['maximo'] - df['maximo'].shift(1)
    down_move = df['minimo'].shift(1) - df['minimo']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr = calcular_atr(df, period)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr)
    suma_di = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / suma_di
    dx = dx.replace([np.inf, -np.inf], np.nan)
    return dx.rolling(period).mean()

def detectar_patrones_smc(df):
    """ Detecta vacíos de liquidez (FVG) y saca niveles exactos de SL y TP basados en mechas """
    if len(df) < 20: return None
    
    # Revisamos las últimas velas buscando el patrón
    for i in range(len(df)-10, len(df)-1):
        
        # SMC ALCISTA (Oportunidad de COMPRA)
        if df['minimo'].iloc[i] > df['maximo'].iloc[i-2]: # Hueco Alcista (FVG)
            if df['minimo'].iloc[-1] <= df['minimo'].iloc[i] and df['cierre'].iloc[-1] >= df['maximo'].iloc[i-2]: # Mitigación
                min_sweep = df['minimo'].iloc[i-15:i].min() # Liquidez barrida
                if df['minimo'].iloc[i-4:i].min() == min_sweep:
                    sl = min_sweep * 0.995 # SL justo debajo de la mecha
                    tp = df['maximo'].iloc[-20:].max() # TP en el máximo reciente
                    return {"tipo": "SMC_ALCISTA", "sl": sl, "tp": tp}
                    
        # SMC BAJISTA (Alerta de CAÍDA / Venta)
        if df['maximo'].iloc[i] < df['minimo'].iloc[i-2]: # Hueco Bajista (FVG)
            if df['maximo'].iloc[-1] >= df['maximo'].iloc[i] and df['cierre'].iloc[-1] <= df['minimo'].iloc[i-2]: # Mitigación
                max_sweep = df['maximo'].iloc[i-15:i].max()
                if df['maximo'].iloc[i-4:i].max() == max_sweep:
                    sl = max_sweep * 1.005 # SL justo arriba de la mecha
                    tp = df['minimo'].iloc[-20:].min()
                    return {"tipo": "SMC_BAJISTA", "sl": sl, "tp": tp}
    return None

# ==========================================================================
# 4. VALIDACIÓN HISTÓRICA
# ==========================================================================
def validar_senal_historica(df, condicion_activa, horizonte=HORIZONTE_VALIDACION):
    disparos = np.where(condicion_activa)[0]
    disparos = disparos[disparos < len(df) - horizonte]
    if len(disparos) < 5:
        return {'n_casos': len(disparos), 'win_rate': None, 'retorno_promedio': None, 'suficiente': False}
    retornos = []
    for idx in disparos:
        precio_entrada = df.iloc[idx]['cierre']
        precio_futuro = df.iloc[idx + horizonte]['cierre']
        retornos.append((precio_futuro / precio_entrada - 1) * 100)
    retornos = np.array(retornos)
    return {
        'n_casos': len(disparos),
        'win_rate': (retornos > 0).mean() * 100,
        'retorno_promedio': retornos.mean(),
        'suficiente': len(disparos) >= 15,
    }

def es_mercado_spot_valido(exchange, simbolo):
    mercado = exchange.markets.get(simbolo)
    if mercado is None: return False
    return mercado.get('spot', False) is True and mercado.get('type', 'spot') == 'spot'

# ==========================================================================
# 5. TELEGRAM Y ESTADO
# ==========================================================================
def enviar_alerta_telegram(mensaje):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Faltan las credenciales de Telegram.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"❌ Excepción en Telegram: {e}")

def cargar_estado():
    if os.path.exists(ESTADO_PATH):
        try:
            with open(ESTADO_PATH, 'r') as f: return json.load(f)
        except Exception: return {}
    return {}

def guardar_estado(estado):
    with open(ESTADO_PATH, 'w') as f: json.dump(estado, f, indent=2)

# ==========================================================================
# 6. ANÁLISIS DE ACTIVOS (NÚCLEO Y CRECIMIENTO) - CON TEXTOS PARA EL EXCEL
# ==========================================================================
def analizar_activo_largo_plazo(exchange, simbolo, temporalidad, descuento_pct, rsi_compra, rsi_venta, sobreprecio_pct, es_crecimiento=False):
    velas = descargar_velas_cerradas(exchange, simbolo, temporalidad, VELAS_ANALISIS)
    if not velas or len(velas) < 60: return None

    df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
    df['RSI'] = calcular_rsi(df['cierre'], period=14)
    df['Media_30'] = df['cierre'].rolling(window=30).mean()
    df['ATR'] = calcular_atr(df, period=14)
    df = df.dropna().reset_index(drop=True)
    if len(df) < 30: return None

    precio = df.iloc[-1]['cierre']
    rsi = df.iloc[-1]['RSI']
    media_30 = df.iloc[-1]['Media_30']
    atr = df.iloc[-1]['ATR']

    condicion_descuento = (df['cierre'] <= df['Media_30'] * (1 - descuento_pct)) | (df['RSI'] < rsi_compra)
    condicion_sobrecompra = df['RSI'] >= rsi_venta

    prefijo_precaucion = "⚠️ *(Alta Volatilidad):* " if es_crecimiento else ""

    # Textos Conversacionales (Venta, Excel, Espera)
    if precio <= (media_30 * (1 - descuento_pct)) or rsi < rsi_compra:
        accion = "COMPRAR"
        etiqueta = f"{prefijo_precaucion}🟢 *COMPRAR:* Activo en descuento. Buen momento para anotar acumulación en tu Excel."
        validacion = validar_senal_historica(df, condicion_descuento.values)
    elif rsi >= rsi_venta:
        accion = "EVALUAR_VENTA"
        etiqueta = f"{prefijo_precaucion}🟡 *TOMAR BENEFICIOS:* Euforia extrema. **Si ya tienes esta moneda, es el momento ideal para VENDER una parte.**"
        validacion = validar_senal_historica(df, condicion_sobrecompra.values)
    elif precio > (media_30 * (1 + sobreprecio_pct)):
        accion = "ESPERAR"
        etiqueta = f"{prefijo_precaucion}🔴 *ESPERAR:* Precio elevado. NO COMPRES AHORA, mantén liquidez."
        validacion = None
    else:
        accion = "NEUTRO"
        etiqueta = f"{prefijo_precaucion}⚪ *ZONA NEUTRA:* Mercado estable. Mantener posiciones."
        validacion = None

    distancia_atr = atr * 2.5
    sugerencia_tamano = (CAPITAL_REFERENCIA * RIESGO_POR_TRADE / distancia_atr) if distancia_atr > 0 else None

    return {
        'simbolo': simbolo, 'precio': precio, 'rsi': rsi, 'media_30': media_30,
        'accion': accion, 'etiqueta': etiqueta, 'validacion': validacion,
        'sugerencia_tamano': sugerencia_tamano, 'atr': atr,
    }

# ==========================================================================
# 7. EJECUCIÓN PRINCIPAL DEL BOT MAESTRO
# ==========================================================================
def ejecutar_bot_maestro():
    exchange = ccxt.kraken()
    exchange.load_markets()

    print("="*70)
    print(" ESCÁNER CUANTITATIVO MAESTRO + SMC FRANCOTIRADOR ")
    print("="*70)

    estado_anterior = cargar_estado()
    estado_nuevo = {}
    alertas_nucleo, alertas_crecimiento, alertas_satelite = [], [], []

    # --- ESCUDO MACRO (BTC Diario) ---
    print("\n🛡️ Verificando Escudo Macro (BTC, diario)...")
    btc_saludable = False
    escudo_verificado = False
    try:
        velas_btc = descargar_velas_cerradas(exchange, 'BTC/USD', TEMPORALIDAD_ESCUDO, VELAS_ESCUDO_BTC)
        if velas_btc:
            df_btc = pd.DataFrame(velas_btc, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
            df_btc['SMA_50'] = df_btc['cierre'].rolling(window=50).mean()
            precio_btc = df_btc.iloc[-1]['cierre']
            sma_50_btc = df_btc.iloc[-1]['SMA_50']
            btc_saludable = precio_btc >= sma_50_btc
            escudo_verificado = True
            print(f"  BTC {'✅ saludable' if btc_saludable else '⚠️ débil'}")
    except Exception as e:
        print(f"⚠️ Error Escudo: {e}")

    # --- NÚCLEO CONSERVADOR ---
    for simbolo in NUCLEO_CONSERVADOR:
        r = analizar_activo_largo_plazo(exchange, simbolo, TEMPORALIDAD_NUCLEO, 0.04, 40, 75, 0.05, False)
        if r:
            alertas_nucleo.append(r)
            estado_nuevo[simbolo] = r['accion']

    # --- NIVEL CRECIMIENTO ---
    for simbolo in NIVEL_CRECIMIENTO:
        r = analizar_activo_largo_plazo(exchange, simbolo, TEMPORALIDAD_CRECIMIENTO, 0.07, 35, 80, 0.08, True)
        if r:
            alertas_crecimiento.append(r)
            estado_nuevo[simbolo] = r['accion']

    # --- SATÉLITE DE ALTO RIESGO (SMC + CUANTITATIVO en 4h) ---
    if btc_saludable:
        print(f"\n⚡ Analizando Satélite de Corto Plazo (4h)...")
        try:
            tickers = exchange.fetch_tickers()
            excluidos = set(NUCLEO_CONSERVADOR) | set(NIVEL_CRECIMIENTO)
            candidatas = [s for s, t in tickers.items() if '/USD' in s and s not in excluidos and 'USDT' not in s and es_mercado_spot_valido(exchange, s) and t.get('quoteVolume', 0) > 1000000]
            
            # Ordenar por volumen para operar las más líquidas y agarrar el top 10
            candidatas = sorted(candidatas, key=lambda s: tickers[s].get('quoteVolume', 0), reverse=True)[:10]

            for simbolo in candidatas:
                velas = descargar_velas_cerradas(exchange, simbolo, TEMPORALIDAD_SATELITE, VELAS_ANALISIS)
                if not velas or len(velas) < 60: continue
                
                df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
                df['RSI'] = calcular_rsi(df['cierre'], period=14)
                _, _, lower = calcular_bollinger_bands(df['cierre'])
                df['BB_Lower'] = lower
                df['Vol_Medio'] = df['volumen'].rolling(window=20).mean()
                df['ATR'] = calcular_atr(df, period=14)
                df['ADX'] = calcular_adx(df, period=14)
                df['EMA_TENDENCIA'] = df['cierre'].ewm(span=200, adjust=False).mean()
                df = df.dropna().reset_index(drop=True)
                
                ultima = df.iloc[-1]
                
                # 1. EVALUAR PATRÓN SMC (Velas de Precio)
                patron_smc = detectar_patrones_smc(df)
                
                # 2. EVALUAR PATRÓN CUANTITATIVO (Indicadores Matemáticos)
                regimen_ok = ultima['ADX'] < SATELITE_ADX_MAX
                tendencia_ok = ultima['cierre'] > ultima['EMA_TENDENCIA']
                cuantitativo_ok = (ultima['RSI'] <= SATELITE_RSI_ENTRADA and 
                                   ultima['cierre'] <= ultima['BB_Lower'] * 1.01 and 
                                   ultima['volumen'] >= ultima['Vol_Medio'] * 0.7 and 
                                   regimen_ok and tendencia_ok)

                # Si alguna de las dos estrategias dispara alerta, la guardamos
                if patron_smc:
                    distancia_riesgo = abs(ultima['cierre'] - patron_smc['sl'])
                    tamano = (CAPITAL_REFERENCIA * RIESGO_POR_TRADE / distancia_riesgo) if distancia_riesgo > 0 else 0
                    alertas_satelite.append({
                        'simbolo': simbolo, 'precio': ultima['cierre'], 'rsi': ultima['RSI'],
                        'tipo_alerta': patron_smc['tipo'], 'tp': patron_smc['tp'], 'sl': patron_smc['sl'],
                        'sugerencia_tamano': tamano, 'validacion': None
                    })
                    estado_nuevo[simbolo] = patron_smc['tipo']
                    
                elif cuantitativo_ok:
                    sl = ultima['cierre'] - (SATELITE_ATR_SL_MULT * ultima['ATR'])
                    tp = ultima['cierre'] + (SATELITE_ATR_TP_MULT * ultima['ATR'])
                    distancia_riesgo = ultima['cierre'] - sl
                    tamano = (CAPITAL_REFERENCIA * RIESGO_POR_TRADE / distancia_riesgo) if distancia_riesgo > 0 else 0
                    
                    cond_entrada = ((df['RSI'] <= SATELITE_RSI_ENTRADA) & (df['cierre'] <= df['BB_Lower'] * 1.01))
                    val = validar_senal_historica(df, cond_entrada.values)
                    
                    alertas_satelite.append({
                        'simbolo': simbolo, 'precio': ultima['cierre'], 'rsi': ultima['RSI'],
                        'tipo_alerta': 'CUANTITATIVO_ALCISTA', 'tp': tp, 'sl': sl,
                        'sugerencia_tamano': tamano, 'validacion': val
                    })
                    estado_nuevo[simbolo] = 'CUANTITATIVO_ALCISTA'
                    
        except Exception as e:
            print(f"⚠️ Error Satélites: {e}")
    else:
        print("\n🚫 Satélites bloqueados (escudo macro no saludable).")

    # --- REPORTE Y FILTRADO DE ALERTAS ---
    def hubo_cambio(simbolo, accion):
        return estado_anterior.get(simbolo) != accion

    def filtrar(lista, clave_estado):
        return [a for a in lista if not SOLO_ALERTAR_CAMBIOS or hubo_cambio(a['simbolo'], a.get('accion', clave_estado))]

    nucleo_enviar = filtrar(alertas_nucleo, None)
    crecimiento_enviar = filtrar(alertas_crecimiento, None)
    
    # Para satélites validamos si el tipo de alerta cambió
    satelite_enviar = [a for a in alertas_satelite if not SOLO_ALERTAR_CAMBIOS or hubo_cambio(a['simbolo'], a['tipo_alerta'])]

    if nucleo_enviar or crecimiento_enviar or satelite_enviar:
        ahora = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        mensaje = f"🚨 *REPORTE DE INVERSIÓN (EXCEL & SMC)* — {ahora}\n\n"
        
        if nucleo_enviar:
            mensaje += "🛡️ *NÚCLEO CONSERVADOR (BTC/ETH)*\n"
            for op in nucleo_enviar:
                mensaje += f"• *{op['simbolo']}* | Precio: `${op['precio']:,.2f}`\n  {op['etiqueta']}\n\n"
                
        if crecimiento_enviar:
            mensaje += "🚀 *CRECIMIENTO Y ALTO POTENCIAL*\n"
            for op in crecimiento_enviar:
                mensaje += f"• *{op['simbolo']}* | Precio: `${op['precio']:,.2f}`\n  {op['etiqueta']}\n\n"
                
        if satelite_enviar:
            mensaje += "🎯 *RADAR DE ALTO RIESGO (4 Horas)*\n"
            for op in satelite_enviar:
                if op['tipo_alerta'] == "SMC_ALCISTA":
                    mensaje += f"🟢 *COMPRA FRANCOTIRADOR (SMC)* | *{op['simbolo']}*\n  _Patrón de velas: Barrido de liquidez y retroceso a FVG._\n"
                elif op['tipo_alerta'] == "SMC_BAJISTA":
                    mensaje += f"🔴 *ALERTA DE CAÍDA (SMC)* | *{op['simbolo']}*\n  _Patrón de velas Bajista. Si estás operando, ajusta tu Stop Loss._\n"
                else:
                    mensaje += f"🟢 *COMPRA CUANTITATIVA* | *{op['simbolo']}*\n  _Sobrevendida matemáticamente (RSI+Bollinger)._\n"
                
                mensaje += (f"  Entrada: `${op['precio']:,.2f}`\n"
                            f"  🎯 TP: `${op['tp']:,.2f}` | 🛑 SL: `${op['sl']:,.2f}`\n"
                            f"  📏 Tamaño sugerido: `{op['sugerencia_tamano']:.4f}` unid.\n\n")

        mensaje += "_Herramienta Híbrida: Cuantitativa + Price Action._"
        enviar_alerta_telegram(mensaje)
    else:
        print("\nℹ️ Sin cambios de estado respecto a la última corrida.")

    guardar_estado(estado_nuevo)

if __name__ == '__main__':
    ejecutar_bot_maestro()
