import ccxt
import pandas as pd
import numpy as np
import os
import json
import requests
from datetime import datetime, timezone

# ==========================================================================
# 1. CONFIGURACIÓN
# ==========================================================================
# --- Núcleo conservador: los más establecidos y líquidos, timeframe diario ---
NUCLEO_CONSERVADOR = ['BTC/USD', 'ETH/USD']
TEMPORALIDAD_NUCLEO = '1d'

# --- Nivel intermedio: mismo horizonte (diario) pero activos más volátiles,
#     por eso usa umbrales algo más anchos (más margen antes de reaccionar) ---
NIVEL_INTERMEDIO = ['SOL/USD']
TEMPORALIDAD_INTERMEDIO = '1d'

# --- Satélite de alto riesgo: corto plazo, recalibrado a 4h.
#     Valores elegidos según convención documentada por la industria/bibliografía
#     (ver justificación de cada uno abajo). Siguen siendo un PUNTO DE PARTIDA
#     razonado, no una garantía — la única prueba real es tu propio walk-forward
#     con datos históricos de 4h. ---
TEMPORALIDAD_SATELITE = '4h'

# RSI de entrada: Wilder (1978) documenta 30 como sobreventa estándar, pero
# múltiples guías de trading cripto (ej. Binance Academy, Babypips) recomiendan
# bajar a 20-25 en activos de alta volatilidad como altcoins, porque el RSI
# estándar de 30 dispara con demasiada frecuencia y genera muchos falsos
# positivos ("ruido") en activos que oscilan tan fuerte. Para el satélite
# (altcoins, el bloque de mayor riesgo) uso 25: más selectivo que el estándar,
# menos extremo que 20.
SATELITE_RSI_ENTRADA = 25

# ADX máximo (filtro de régimen): Wilder define <20 como "ausencia de
# tendencia" (rango puro) y 20-25 como zona ambigua. Para el bloque de mayor
# riesgo conviene la definición más estricta (20), no la más permisiva (25),
# precisamente porque aquí es donde más cuesta un falso positivo (comprar una
# caída que en realidad es tendencia bajista, no rango).
SATELITE_ADX_MAX = 20

# Multiplicador ATR del Stop Loss: Van Tharp documenta 1.5x-3x como rango
# profesional. 2.0x es el punto medio y el valor por defecto más citado en
# frameworks de trading algorítmico (ej. backtesting.py, Freqtrade usan 2x
# ATR como default sugerido para stops en timeframes intradía/4h).
SATELITE_ATR_SL_MULT = 2.0

# Take Profit: se fija en 4x ATR para lograr un ratio riesgo:beneficio de 2:1.
# Esto no es arbitrario — es uno de los principios más repetidos en la
# literatura de gestión de riesgo (Van Tharp, Alexander Elder en "Trading for
# a Living"): con R:R de 2:1, el sistema puede ser rentable incluso con win
# rate tan bajo como ~35-40%, lo cual da más margen de error que un R:R de
# 1.5:1 (que necesita win rate más alto para ser rentable).
SATELITE_ATR_TP_MULT = 4.0

# Filtro de tendencia mayor: EMA de 200 periodos es, con diferencia, el nivel
# más citado y más vigilado en trading de cripto específicamente (a diferencia
# del Golden Cross SMA50/200, que es más una convención de acciones/forex).
# Al ser un nivel que gran parte del mercado observa, tiene más probabilidad
# de actuar como soporte/resistencia autocumplida en cripto.
SATELITE_FILTRO_TENDENCIA = 'ema200'

VELAS_ANALISIS = 300
VELAS_ESCUDO_BTC = 80
TEMPORALIDAD_ESCUDO = '1d'             # el escudo macro siempre es diario, independiente de lo demás

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

ESTADO_PATH = 'bot_estado.json'
SOLO_ALERTAR_CAMBIOS = True

CAPITAL_REFERENCIA = 370000
RIESGO_POR_TRADE = 0.01
HORIZONTE_VALIDACION = 14


# ==========================================================================
# 2. DESCARGA DE VELAS — SIEMPRE DESCARTA LA VELA EN FORMACIÓN
#    Esto es lo que permite correr el bot cada hora sin repintado: sin
#    importar cuántas veces al día lo ejecutes, el análisis diario siempre
#    usa la última vela DIARIA ya cerrada, nunca la que sigue actualizándose.
# ==========================================================================
def descargar_velas_cerradas(exchange, simbolo, temporalidad, limit):
    tf_ms = exchange.parse_timeframe(temporalidad) * 1000
    velas = exchange.fetch_ohlcv(simbolo, timeframe=temporalidad, limit=limit + 1)
    if not velas:
        return velas
    ahora = exchange.milliseconds()
    ultima_vela_ts = velas[-1][0]
    if ultima_vela_ts + tf_ms > ahora:
        velas = velas[:-1]   # descarta la vela todavía en formación
    return velas[-limit:] if len(velas) > limit else velas


# ==========================================================================
# 3. INDICADORES
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


# ==========================================================================
# 4. VALIDACIÓN HISTÓRICA DE UNA SEÑAL
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
    if mercado is None:
        return False
    return mercado.get('spot', False) is True and mercado.get('type', 'spot') == 'spot'


# ==========================================================================
# 5. TELEGRAM Y ESTADO
# ==========================================================================
def enviar_alerta_telegram(mensaje):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Faltan las credenciales de Telegram.")
        print(mensaje)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=15)
        print("📱 Alerta enviada." if response.status_code == 200 else f"❌ Error Telegram: {response.text}")
    except Exception as e:
        print(f"❌ Excepción en Telegram: {e}")


def cargar_estado():
    if os.path.exists(ESTADO_PATH):
        try:
            with open(ESTADO_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def guardar_estado(estado):
    with open(ESTADO_PATH, 'w') as f:
        json.dump(estado, f, indent=2)


# ==========================================================================
# 6. ANÁLISIS DE UN ACTIVO DE "MEDIANO/LARGO PLAZO" (núcleo o intermedio)
#    Misma lógica, umbrales configurables por nivel para reflejar distinta
#    volatilidad esperada (SOL necesita más margen que BTC/ETH).
# ==========================================================================
def analizar_activo_largo_plazo(exchange, simbolo, temporalidad, descuento_pct, rsi_compra, rsi_venta, sobreprecio_pct):
    velas = descargar_velas_cerradas(exchange, simbolo, temporalidad, VELAS_ANALISIS)
    if not velas or len(velas) < 60:
        return None

    df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
    df['RSI'] = calcular_rsi(df['cierre'], period=14)
    df['Media_30'] = df['cierre'].rolling(window=30).mean()
    df['ATR'] = calcular_atr(df, period=14)
    df = df.dropna().reset_index(drop=True)
    if len(df) < 30:
        return None

    precio = df.iloc[-1]['cierre']
    rsi = df.iloc[-1]['RSI']
    media_30 = df.iloc[-1]['Media_30']
    atr = df.iloc[-1]['ATR']

    condicion_descuento = (df['cierre'] <= df['Media_30'] * (1 - descuento_pct)) | (df['RSI'] < rsi_compra)
    condicion_sobrecompra = df['RSI'] >= rsi_venta

    if precio <= (media_30 * (1 - descuento_pct)) or rsi < rsi_compra:
        accion = "COMPRAR"
        etiqueta = "🟢 *COMPRAR* (activo en descuento por corrección)"
        validacion = validar_senal_historica(df, condicion_descuento.values)
    elif rsi >= rsi_venta:
        accion = "EVALUAR_VENTA"
        etiqueta = "🟡 *EVALUAR TOMA DE BENEFICIOS* (sobrecalentamiento)"
        validacion = validar_senal_historica(df, condicion_sobrecompra.values)
    elif precio > (media_30 * (1 + sobreprecio_pct)):
        accion = "ESPERAR"
        etiqueta = "🔴 *ESPERAR* (precio por encima del promedio, mantén liquidez)"
        validacion = None
    else:
        accion = "NEUTRO"
        etiqueta = "⚪ *ZONA NEUTRA*"
        validacion = None

    distancia_atr = atr * 2.5
    sugerencia_tamano = (CAPITAL_REFERENCIA * RIESGO_POR_TRADE / distancia_atr) if distancia_atr > 0 else None

    return {
        'simbolo': simbolo, 'precio': precio, 'rsi': rsi, 'media_30': media_30,
        'accion': accion, 'etiqueta': etiqueta, 'validacion': validacion,
        'sugerencia_tamano': sugerencia_tamano, 'atr': atr,
    }


# ==========================================================================
# 7. BOT MAESTRO
# ==========================================================================
def ejecutar_bot_maestro():
    exchange = ccxt.kraken()
    exchange.load_markets()

    print("="*70)
    print(" ESCÁNER CUANTITATIVO MAESTRO (NÚCLEO / INTERMEDIO / SATÉLITE)")
    print("="*70)

    estado_anterior = cargar_estado()
    estado_nuevo = {}
    alertas_nucleo, alertas_intermedio, alertas_satelite = [], [], []

    # --- ESCUDO MACRO (siempre diario, fail-safe) ---
    print("\n🛡️ Verificando Escudo Macro (BTC, diario)...")
    btc_saludable = False
    escudo_verificado = False
    try:
        velas_btc = descargar_velas_cerradas(exchange, 'BTC/USD', TEMPORALIDAD_ESCUDO, VELAS_ESCUDO_BTC)
        if velas_btc and len(velas_btc) >= 50:
            df_btc = pd.DataFrame(velas_btc, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
            df_btc['SMA_50'] = df_btc['cierre'].rolling(window=50).mean()
            precio_btc = df_btc.iloc[-1]['cierre']
            sma_50_btc = df_btc.iloc[-1]['SMA_50']
            btc_saludable = precio_btc >= sma_50_btc
            escudo_verificado = True
            print(f"  BTC {'✅ saludable' if btc_saludable else '⚠️ débil'}: {precio_btc:.2f} vs SMA50 {sma_50_btc:.2f}")
    except Exception as e:
        print(f"⚠️ Error al verificar el escudo macro: {e}")

    if not escudo_verificado:
        btc_saludable = False
        print("  -> No verificable: satélites bloqueados por seguridad.")

    # --- NÚCLEO CONSERVADOR (BTC/ETH, diario, umbrales estándar) ---
    print("\n🛡️ Analizando Núcleo Conservador (BTC/ETH)...")
    for simbolo in NUCLEO_CONSERVADOR:
        try:
            r = analizar_activo_largo_plazo(
                exchange, simbolo, TEMPORALIDAD_NUCLEO,
                descuento_pct=0.04, rsi_compra=40, rsi_venta=75, sobreprecio_pct=0.05
            )
            if r:
                alertas_nucleo.append(r)
                estado_nuevo[simbolo] = r['accion']
        except Exception as e:
            print(f"⚠️ Error en Núcleo {simbolo}: {e}")

    # --- NIVEL INTERMEDIO (SOL, diario, umbrales más anchos por su volatilidad) ---
    print("\n🟠 Analizando Nivel Intermedio (SOL)...")
    for simbolo in NIVEL_INTERMEDIO:
        try:
            r = analizar_activo_largo_plazo(
                exchange, simbolo, TEMPORALIDAD_INTERMEDIO,
                descuento_pct=0.07, rsi_compra=35, rsi_venta=80, sobreprecio_pct=0.08
            )
            if r:
                alertas_intermedio.append(r)
                estado_nuevo[simbolo] = r['accion']
        except Exception as e:
            print(f"⚠️ Error en Intermedio {simbolo}: {e}")

    # --- SATÉLITE (4h, dinámico, filtro ADX de régimen) ---
    if btc_saludable:
        print(f"\n🚀 Analizando Satélite ({TEMPORALIDAD_SATELITE}, con filtro de régimen ADX)...")
        try:
            tickers = exchange.fetch_tickers()
            excluidos = set(NUCLEO_CONSERVADOR) | set(NIVEL_INTERMEDIO)
            candidatas = []
            for s, ticker in tickers.items():
                if '/USD' not in s or s in excluidos or 'USDT' in s or 'USDC' in s:
                    continue
                if not es_mercado_spot_valido(exchange, s):
                    continue
                vol = ticker.get('quoteVolume', 0)
                if vol and vol > 1_000_000:
                    candidatas.append((s, vol))
            candidatas.sort(key=lambda x: x[1], reverse=True)
            top_altcoins = [item[0] for item in candidatas[:5]]

            for simbolo in top_altcoins:
                try:
                    velas = descargar_velas_cerradas(exchange, simbolo, TEMPORALIDAD_SATELITE, VELAS_ANALISIS)
                    if not velas or len(velas) < 60:
                        continue
                    df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
                    df['RSI'] = calcular_rsi(df['cierre'], period=14)
                    _, _, lower = calcular_bollinger_bands(df['cierre'])
                    df['BB_Lower'] = lower
                    df['Vol_Medio'] = df['volumen'].rolling(window=20).mean()
                    df['ATR'] = calcular_atr(df, period=14)
                    df['ADX'] = calcular_adx(df, period=14)
                    df['EMA_TENDENCIA'] = df['cierre'].ewm(span=200, adjust=False).mean()
                    df = df.dropna().reset_index(drop=True)
                    if len(df) < 30:
                        continue

                    ultima = df.iloc[-1]
                    regimen_ok = ultima['ADX'] < SATELITE_ADX_MAX
                    tendencia_ok = ultima['cierre'] > ultima['EMA_TENDENCIA']
                    condicion_entrada = (
                        (df['RSI'] <= SATELITE_RSI_ENTRADA) &
                        (df['cierre'] <= df['BB_Lower'] * 1.01) &
                        (df['volumen'] >= df['Vol_Medio'] * 0.7) &
                        (df['ADX'] < SATELITE_ADX_MAX)
                    )

                    if (ultima['RSI'] <= SATELITE_RSI_ENTRADA and ultima['cierre'] <= ultima['BB_Lower'] * 1.01
                            and ultima['volumen'] >= ultima['Vol_Medio'] * 0.7 and regimen_ok and tendencia_ok):

                        sl = ultima['cierre'] - (SATELITE_ATR_SL_MULT * ultima['ATR'])
                        tp = ultima['cierre'] + (SATELITE_ATR_TP_MULT * ultima['ATR'])
                        validacion = validar_senal_historica(df, condicion_entrada.values)
                        distancia_riesgo = ultima['cierre'] - sl
                        sugerencia_tamano = (CAPITAL_REFERENCIA * RIESGO_POR_TRADE / distancia_riesgo) if distancia_riesgo > 0 else None

                        alertas_satelite.append({
                            'simbolo': simbolo, 'precio': ultima['cierre'], 'rsi': ultima['RSI'],
                            'tp': tp, 'sl': sl, 'validacion': validacion, 'sugerencia_tamano': sugerencia_tamano,
                        })
                        estado_nuevo[simbolo] = 'SATELITE_ENTRADA'
                except Exception as e:
                    print(f"⚠️ Error en satélite {simbolo}: {e}")
        except Exception as e:
            print(f"⚠️ Error obteniendo tickers: {e}")
    else:
        print("\n🚫 Satélites bloqueados (escudo macro no saludable o no verificable).")

    # --- REPORTE ---
    def hubo_cambio(simbolo, accion):
        return estado_anterior.get(simbolo) != accion

    def filtrar(lista, clave_estado):
        return [a for a in lista if not SOLO_ALERTAR_CAMBIOS or hubo_cambio(a['simbolo'], a.get('accion', clave_estado))]

    nucleo_enviar = filtrar(alertas_nucleo, None)
    intermedio_enviar = filtrar(alertas_intermedio, None)
    satelite_enviar = [a for a in alertas_satelite if not SOLO_ALERTAR_CAMBIOS or hubo_cambio(a['simbolo'], 'SATELITE_ENTRADA')]

    def bloque_texto(titulo, lista, es_satelite=False):
        texto = f"{titulo}\n"
        for op in lista:
            if es_satelite:
                texto += (f"• *{op['simbolo']}* | Entrada: `${op['precio']:,.2f}` | RSI: `{op['rsi']:.1f}`\n"
                          f"  🎯 TP: `${op['tp']:,.2f}` | 🛑 SL: `${op['sl']:,.2f}`\n")
            else:
                texto += f"• *{op['simbolo']}* | Precio: `${op['precio']:,.2f}` | RSI: `{op['rsi']:.1f}`\n  {op['etiqueta']}\n"
            v = op['validacion']
            if v and v['suficiente']:
                texto += f"  📊 Histórico ({v['n_casos']} casos): win rate {v['win_rate']:.0f}%, retorno prom. {v['retorno_promedio']:+.1f}%\n"
            elif v and v['n_casos'] > 0:
                texto += f"  ⚠️ Solo {v['n_casos']} casos históricos — poco confiable\n"
            if op.get('sugerencia_tamano'):
                texto += f"  📏 Tamaño orientativo: `{op['sugerencia_tamano']:.4f}` unidades\n"
            texto += "\n"
        return texto

    if nucleo_enviar or intermedio_enviar or satelite_enviar:
        ahora = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        mensaje = f"🚨 *REPORTE CUANTITATIVO* — {ahora}\n\n"
        if nucleo_enviar:
            mensaje += bloque_texto("🛡️ *NÚCLEO CONSERVADOR (BTC/ETH)*", nucleo_enviar)
        if intermedio_enviar:
            mensaje += bloque_texto("🟠 *NIVEL INTERMEDIO (SOL)*", intermedio_enviar)
        if satelite_enviar:
            mensaje += bloque_texto("🚀 *SATÉLITE (alto riesgo, 4h)*", satelite_enviar, es_satelite=True)
        mensaje += ("_Herramienta de apoyo con validación histórica limitada. "
                    "No es asesoría financiera personalizada._")
        enviar_alerta_telegram(mensaje)
    else:
        print("\nℹ️ Sin cambios de estado respecto a la última corrida.")

    guardar_estado(estado_nuevo)


if __name__ == '__main__':
    ejecutar_bot_maestro()
