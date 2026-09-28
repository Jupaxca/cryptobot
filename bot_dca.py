from datetime import datetime, timezone
import json
import os
import time
import warnings
import ccxt
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')

# ==========================================================================
# 1. CONFIGURACIÓN GENERAL DEL PORTAFOLIO (NARRATIVAS 2026)
# ==========================================================================
NUCLEO_CONSERVADOR = [
    'BTC/USDT',
    'ETH/USDT',
    'SOL/USDT',
    'BNB/USDT',
    'LINK/USDT',
]
TEMPORALIDAD_NUCLEO = '1d'

NIVEL_CRECIMIENTO = [
    'NEAR/USDT',
    'TAO/USDT',
    'SEI/USDT',
    'ADA/USDT',
    'AVAX/USDT',
    'RENDER/USDT',
    'FET/USDT',
    'TIA/USDT',
    'ARB/USDT',
]
TEMPORALIDAD_CRECIMIENTO = '1d'

SEGUIMIENTO_ESTRATEGICO = [
    'XRP/USDT',
    'SUI/USDT',
    'APT/USDT',
    'ONDO/USDT',
    'OM/USDT',
    'HNT/USDT',
    'FIL/USDT',
    'PENDLE/USDT',
]
TEMPORALIDAD_SEGUIMIENTO = '1d'

ALTO_RIESGO = [
    'PEPE/USDT',
    'WIF/USDT',
    'POPCAT/USDT',
    'KAS/USDT',
    'THETA/USDT',
    'ASTR/USDT',
]
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


def calcular_cvd(df):
  """Inferencia Estadística de Presión de Ballenas (CVD)"""
  rango = df['maximo'] - df['minimo']
  rango = rango.replace(0, 0.0001)

  fuerza_compradora = ((df['cierre'] - df['minimo']) / rango) * df['volumen']
  fuerza_vendedora = df['volumen'] - fuerza_compradora

  df['Delta_Volumen'] = fuerza_compradora - fuerza_vendedora
  df['CVD'] = df['Delta_Volumen'].cumsum()
  return df


def calcular_kelly_fraccional(
    prob_exito_pct, tp_precio, sl_precio, precio_entrada, fraccion=0.5
):
  """Criterio de Kelly matemático para gestión de riesgo óptimo"""
  if prob_exito_pct is None:
    return RIESGO_POR_TRADE

  p = prob_exito_pct / 100.0
  q = 1.0 - p

  distancia_tp = abs(tp_precio - precio_entrada)
  distancia_sl = abs(precio_entrada - sl_precio)
  if distancia_sl == 0:
    return 0.0

  b = distancia_tp / distancia_sl
  kelly_pct = p - (q / b)

  if kelly_pct <= 0:
    return 0.0

  riesgo_final = kelly_pct * fraccion
  return max(min(riesgo_final, 0.05), 0.005)


def estimar_pico_confluencia(df, horizonte_velas=HORIZONTE_VALIDACION):
  """Combina estadística de Momentum y Machine Learning (Regressor) para calcular el día/vela óptimo de Hold"""
  n = len(df)
  if n < 100:
    return 7, 'Hold ~7 velas (Faltan datos históricos)'

  # 1. CEREBRO ESTADÍSTICO
  dias_al_pico_hist = []
  for i in range(max(0, n - 120 - horizonte_velas), n - horizonte_velas):
    ventana_futura = df['maximo'].iloc[i + 1 : i + 1 + horizonte_velas].values
    if len(ventana_futura) == horizonte_velas:
      dias_al_pico_hist.append(np.argmax(ventana_futura) + 1)

  dia_base_estadistico = (
      int(np.median(dias_al_pico_hist)) if dias_al_pico_hist else 7
  )
  ultima = df.iloc[-1]

  if ultima['RSI'] > 70:
    dia_estadistico = max(1, dia_base_estadistico - 3)
  elif ultima['ADX'] > 28 and ultima['Pendiente_Reg'] > 0:
    dia_estadistico = min(horizonte_velas, dia_base_estadistico + 2)
  else:
    dia_estadistico = dia_base_estadistico

  # 2. CEREBRO IA (XGBoost Regressor)
  dia_ia = dia_estadistico
  confianza_ia = False

  try:
    df_ml = df.copy()
    indexer = pd.api.indexers.FixedForwardWindowIndexer(
        window_size=horizonte_velas
    )
    df_ml['dia_pico_futuro'] = df_ml['maximo'].rolling(window=indexer).apply(
        lambda x: np.argmax(x) + 1, raw=True
    )

    features = [
        'RSI',
        'ATR',
        'ADX',
        'volumen',
        'Hurst',
        'Pendiente_Reg',
        'CRT_Valida',
        'CVD',
        'Delta_Volumen',
    ]
    df_ml = df_ml.dropna(subset=features + ['dia_pico_futuro'])

    X = df_ml[:-horizonte_velas][features]
    y = df_ml[:-horizonte_velas]['dia_pico_futuro']

    if len(X) >= 40:
      modelo_dias = XGBRegressor(
          n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42
      )
      modelo_dias.fit(X, y)

      X_actual = df_ml.iloc[-1:][features]
      prediccion = modelo_dias.predict(X_actual)[0]
      dia_ia = min(max(int(round(prediccion)), 1), horizonte_velas)
      confianza_ia = True
  except Exception:
    pass

  # 3. CONFLUENCIA (Sinergia de Modelos)
  dia_final = int(round((dia_estadistico + dia_ia) / 2))
  diferencia = abs(dia_estadistico - dia_ia)

  if confianza_ia:
    if diferencia <= 2:
      estatus = (
          f'Alta Confianza (Modelos Sincronizados: Est. {dia_estadistico} | IA'
          f' {dia_ia})'
      )
    elif diferencia >= 6:
      estatus = (
          f'Volatilidad (Discrepancia: Est. {dia_estadistico} vs IA {dia_ia})'
      )
    else:
      estatus = f'Promedio calculado (Est. {dia_estadistico} | IA {dia_ia})'
  else:
    estatus = 'Análisis Estadístico (IA sin datos suficientes)'

  recomendacion_hold = f'Hold ~{dia_final} velas | {estatus}'
  return dia_final, recomendacion_hold


def calcular_rsi(series, period=14):
  delta = series.diff()
  gain = delta.where(delta > 0, 0).rolling(window=period).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
  rs = gain / loss
  return (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], 100)


def calcular_vwma(df, period=30):
  return (df['cierre'] * df['volumen']).rolling(period).sum() / df[
      'volumen'
  ].rolling(period).sum()


def calcular_bollinger_bands(series, period=20, std_dev=2):
  middle = series.rolling(window=period).mean()
  std = series.rolling(window=period).std()
  return middle + (std * std_dev), middle, middle - (std * std_dev)


def calcular_atr(df, period=14):
  tr = pd.concat(
      [
          df['maximo'] - df['minimo'],
          (df['maximo'] - df['cierre'].shift(1)).abs(),
          (df['minimo'] - df['cierre'].shift(1)).abs(),
      ],
      axis=1,
  ).max(axis=1)
  return tr.rolling(period).mean()


def calcular_adx(df, period=14):
  up_move = df['maximo'] - df['maximo'].shift(1)
  down_move = df['minimo'].shift(1) - df['minimo']
  plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
  minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
  atr = calcular_atr(df, period)
  plus_di = 100 * (
      pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
  )
  minus_di = 100 * (
      pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
  )
  dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
  return dx.replace([np.inf, -np.inf], np.nan).rolling(period).mean()


def calcular_hurst_vectorizado(series, window=100, max_lag=20):
  hursts = []
  vals = series.values
  for i in range(len(vals)):
    if i < window:
      hursts.append(0.5)
      continue
    sub = vals[i - window : i]
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
    y = vals[i - window : i]
    x = np.arange(window)
    if np.std(y) == 0:
      slopes.append(0.0)
      r2s.append(0.0)
    else:
      slope, _ = np.polyfit(x, y, 1)
      corr = np.corrcoef(x, y)[0, 1]
      slopes.append(slope)
      r2s.append(corr**2 if not np.isnan(corr) else 0.0)
  return pd.Series(slopes, index=series.index), pd.Series(
      r2s, index=series.index
  )


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

    if (
        cierre_penultima > max_estructura
        and vol_penultima > vol_media_penultima
    ):
      senales_alcistas[i] = True
    if (
        cierre_penultima < min_estructura
        and vol_penultima > vol_media_penultima
    ):
      senales_bajistas[i] = True

  if senales_alcistas[-1]:
    sl = df['minimo'].iloc[-10:].min() * 0.995
    tp = df['cierre'].iloc[-1] + (df['cierre'].iloc[-1] - sl) * 2.0
    ultima_senal = {'tipo': 'SMC_ALCISTA', 'sl': sl, 'tp': tp}
  elif senales_bajistas[-1]:
    sl = df['maximo'].iloc[-10:].max() * 1.005
    tp = df['cierre'].iloc[-1] - (sl - df['cierre'].iloc[-1]) * 2.0
    ultima_senal = {'tipo': 'SMC_BAJISTA', 'sl': sl, 'tp': tp}

  return ultima_senal, senales_alcistas, senales_bajistas


def detectar_zonas_rechazo_kmeans(df, n_clusters=5):
  precios = np.concatenate(
      [df['maximo'].values, df['minimo'].values]
  ).reshape(-1, 1)
  kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
  kmeans.fit(precios)
  return sorted(kmeans.cluster_centers_.flatten())


def proyectar_meta_temporal(df, temporalidad, horizonte_velas=HORIZONTE_VALIDACION):
  ultima = df.iloc[-1]
  precio_actual = ultima['cierre']
  atr = ultima['ATR']
  slope = ultima['Pendiente_Reg']

  desplazamiento_esperado = (slope * horizonte_velas) + (atr * 1.5)
  precio_proyectado = max(
      precio_actual + desplazamiento_esperado, precio_actual * 1.01
  )
  variacion_pct = ((precio_proyectado / precio_actual) - 1) * 100

  if temporalidad == '1d':
    tiempo_texto = f'{horizonte_velas} días'
  elif temporalidad == '4h':
    horas = horizonte_velas * 4
    dias = horas / 24
    tiempo_texto = f'{horas}h (~{dias:.1f} días)'
  else:
    tiempo_texto = f'{horizonte_velas} velas'

  return tiempo_texto, precio_proyectado, variacion_pct


def validar_senal_historica(df, condicion_activa, horizonte=HORIZONTE_VALIDACION):
  disparos = np.where(condicion_activa)[0]
  disparos = disparos[disparos < len(df) - horizonte]
  if len(disparos) < 5:
    return {
        'n_casos': len(disparos),
        'win_rate': None,
        'retorno_promedio': None,
        'suficiente': False,
        'mc_confianza': None,
    }

  costo_total = (COMISION_EXCHANGE + SLIPPAGE) * 2
  retornos = np.array([
      (
          (df.iloc[i + horizonte]['cierre'] / df.iloc[i]['cierre'])
          - 1
          - costo_total
      )
      * 100
      for i in disparos
  ])

  np.random.seed(42)
  simulaciones_mc = [
      np.random.choice(retornos, size=len(retornos), replace=True).sum()
      for _ in range(1000)
  ]
  prob_positiva_mc = (np.array(simulaciones_mc) > 0).mean() * 100
  return {
      'n_casos': len(disparos),
      'win_rate': (retornos > 0).mean() * 100,
      'retorno_promedio': retornos.mean(),
      'suficiente': len(disparos) >= 15,
      'mc_confianza': prob_positiva_mc,
  }


def es_mercado_spot_valido(exchange, simbolo):
  mercado = exchange.markets.get(simbolo)
  return (
      mercado
      and mercado.get('spot', False) is True
      and mercado.get('type', 'spot') == 'spot'
  )


# ==========================================================================
# 3. MÓDULOS DE MEMORIA E IA AISLADA
# ==========================================================================
def registrar_prediccion(
    simbolo,
    precio,
    rsi,
    atr,
    adx,
    vol,
    hurst,
    slope,
    r2,
    crt,
    cvd,
    delta_vol,
    prob_subida,
):
  nueva_fila = pd.DataFrame([{
      'timestamp': datetime.now(timezone.utc).isoformat(),
      'simbolo': simbolo,
      'precio_entrada': precio,
      'RSI': rsi,
      'ATR': atr,
      'ADX': adx,
      'volumen': vol,
      'Hurst': hurst,
      'Pendiente_Reg': slope,
      'R2_Tendencia': r2,
      'CRT_Valida': crt,
      'CVD': cvd,
      'Delta_Volumen': delta_vol,
      'prob_subida_predicha': prob_subida,
      'precio_futuro': np.nan,
      'exito_real': np.nan,
  }])

  if not os.path.exists(ARCHIVO_MEMORIA) or os.path.getsize(ARCHIVO_MEMORIA) == 0:
    nueva_fila.to_csv(ARCHIVO_MEMORIA, index=False)
    return

  try:
    df_mem = pd.read_csv(ARCHIVO_MEMORIA)
    if df_mem.empty:
      nueva_fila.to_csv(ARCHIVO_MEMORIA, index=False)
      return

    df_mem['timestamp_dt'] = pd.to_datetime(df_mem['timestamp'])
    ahora_utc = datetime.now(timezone.utc)

    simbolo_reciente = df_mem[
        (df_mem['simbolo'] == simbolo)
        & ((ahora_utc - df_mem['timestamp_dt']).dt.total_seconds() < 12 * 3600)
    ]

    if simbolo_reciente.empty:
      nueva_fila.to_csv(ARCHIVO_MEMORIA, mode='a', header=False, index=False)
  except Exception:
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
        fecha_objetivo = fila['timestamp'] + pd.Timedelta(
            days=horizonte_dias
        )
        since_ms = int(fecha_objetivo.timestamp() * 1000)
        velas = exchange.fetch_ohlcv(
            fila['simbolo'], timeframe='1d', since=since_ms, limit=2
        )
        if velas:
          precio_futuro = velas[0][4]
          df_memoria.at[idx, 'precio_futuro'] = precio_futuro
          df_memoria.at[idx, 'exito_real'] = (
              1 if precio_futuro > fila['precio_entrada'] else 0
          )
          cambios = True
        time.sleep(0.2)
      except Exception:
        pass
  if cambios:
    df_memoria.to_csv(ARCHIVO_MEMORIA, index=False)


def calcular_riesgo_dinamico(riesgo_base=0.01):
  if not os.path.exists(ARCHIVO_MEMORIA):
    return riesgo_base, 'SIN DATOS'
  try:
    if os.path.getsize(ARCHIVO_MEMORIA) == 0:
      return riesgo_base, 'SIN DATOS'
    df_mem = pd.read_csv(ARCHIVO_MEMORIA)
    if df_mem.empty:
      return riesgo_base, 'SIN DATOS'
  except Exception:
    return riesgo_base, 'SIN DATOS'

  auditadas = df_mem.dropna(subset=['exito_real'])
  if len(auditadas) < MIN_CASOS_RIESGO_DINAMICO:
    return (
        riesgo_base,
        f'RECOPILANDO ({len(auditadas)}/{MIN_CASOS_RIESGO_DINAMICO})',
    )
  win_rate_reciente = auditadas.tail(MIN_CASOS_RIESGO_DINAMICO)[
      'exito_real'
  ].mean()
  if win_rate_reciente < 0.40:
    return riesgo_base * 0.5, f'PERDEDORA ({win_rate_reciente*100:.0f}%)'
  elif win_rate_reciente > 0.60:
    return riesgo_base * 1.5, f'GANADORA ({win_rate_reciente*100:.0f}%)'
  return riesgo_base, f'ESTABLE ({win_rate_reciente*100:.0f}%)'


def detectar_regimen_kmeans(df_btc):
  if len(df_btc) < 100:
    return 'DESCONOCIDO'
  try:
    data = pd.DataFrame(index=df_btc.index)
    data['rendimiento'] = df_btc['cierre'].pct_change()
    data['volatilidad'] = df_btc['maximo'] - df_btc['minimo']
    data['tendencia'] = (
        df_btc['cierre'] / df_btc['cierre'].rolling(50).mean() - 1
    )
    data = data.dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data['cluster'] = kmeans.fit_predict(X_scaled)
    medias = data.groupby('cluster')['tendencia'].mean().sort_values()
    mapa_regimenes = {
        medias.index[0]: 'BAJISTA',
        medias.index[1]: 'LATERAL',
        medias.index[2]: 'ALCISTA',
    }
    return mapa_regimenes[data['cluster'].iloc[-1]]
  except Exception:
    return 'LATERAL'


def inferir_probabilidad_xgboost(df, simbolo_actual):
  try:
    df_ml = df.copy()
    df_ml['retorno_futuro'] = (
        df_ml['cierre'].shift(-HORIZONTE_VALIDACION) / df_ml['cierre'] - 1
    )
    df_ml['exito'] = (df_ml['retorno_futuro'] > 0).astype(int)

    features = [
        'RSI',
        'ATR',
        'ADX',
        'volumen',
        'Hurst',
        'Pendiente_Reg',
        'R2_Tendencia',
        'CRT_Valida',
        'CVD',
        'Delta_Volumen',
    ]
    df_ml = df_ml.dropna(subset=features)

    X = df_ml[:-HORIZONTE_VALIDACION][features]
    y = df_ml[:-HORIZONTE_VALIDACION]['exito']

    if os.path.exists(ARCHIVO_MEMORIA):
      try:
        if os.path.getsize(ARCHIVO_MEMORIA) > 0:
          df_mem = pd.read_csv(ARCHIVO_MEMORIA)
          df_mem = df_mem[
              (df_mem['simbolo'] == simbolo_actual)
              & (df_mem['exito_real'].notna())
          ]
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
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
    )
    xgb_model.fit(X, y)

    X_actual = df_ml.iloc[-1:][features]
    probabilidades = xgb_model.predict_proba(X_actual)[0]
    return probabilidades[1] * 100, probabilidades[0] * 100
  except Exception:
    return None, None


# ==========================================================================
# 4. TELEGRAM (TEXTO PLANO Y FRAGMENTADO)
# ==========================================================================
def enviar_alerta_telegram(mensaje):
  if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print(
        '❌ ERROR: TELEGRAM_TOKEN o TELEGRAM_CHAT_ID no están definidos en el'
        ' entorno.'
    )
    return

  mensaje_plano = mensaje.replace('*', '').replace('`', '')
  max_length = 4000
  mensajes = [
      mensaje_plano[i : i + max_length]
      for i in range(0, len(mensaje_plano), max_length)
  ]

  for idx, msg_chunk in enumerate(mensajes):
    try:
      url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
      payload = {
          'chat_id': TELEGRAM_CHAT_ID,
          'text': msg_chunk,
          'parse_mode': '',
      }
      response = requests.post(url, json=payload)
      if response.status_code == 200:
        print(f'📱 Parte {idx+1} enviada con éxito a Telegram.')
      else:
        print(f'❌ Error Telegram HTTP {response.status_code}: {response.text}')
    except Exception as e:
      print(f'❌ Excepción al enviar alerta a Telegram: {e}')


# ==========================================================================
# 5. MOTOR PRINCIPAL DE EJECUCIÓN (ESCENARIO Y PORTAFOLIO)
# ==========================================================================
def ejecutar_bot_cripto():
  print('🚀 INICIANDO ANÁLISIS CUANTITATIVO DE CRIPTOMONEDAS...')
  exchange = ccxt.binance({
      'enableRateLimit': True,
      'options': {'defaultType': 'spot'},
  })
  exchange.load_markets()

  auditar_memoria(exchange)
  riesgo_actual, estado_riesgo = calcular_riesgo_dinamico(RIESGO_POR_TRADE)

  # Descargar escudo BTC
  velas_btc = descargar_velas_cerradas(
      exchange, 'BTC/USDT', TEMPORALIDAD_ESCUDO, VELAS_ESCUDO_BTC
  )
  regimen_btc = 'LATERAL'
  if velas_btc:
    df_btc = pd.DataFrame(
        velas_btc, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen']
    )
    regimen_btc = detectar_regimen_kmeans(df_btc)

  print(f'🛡️ Régimen de Mercado detectado (BTC): {regimen_btc}')
  print(f'📊 Gestión de Riesgo Dinámica: {estado_riesgo}')

  portafolios = [
      ('NÚCLEO CONSERVADOR', NUCLEO_CONSERVADOR, TEMPORALIDAD_NUCLEO),
      ('CRECIMIENTO', NIVEL_CRECIMIENTO, TEMPORALIDAD_CRECIMIENTO),
      ('SEGUIMIENTO ESTRATÉGICO', SEGUIMIENTO_ESTRATEGICO, TEMPORALIDAD_SEGUIMIENTO),
      ('ALTO RIESGO', ALTO_RIESGO, TEMPORALIDAD_ALTO_RIESGO),
  ]

  reporte_general = f'🤖 *REPORTE CUANTITATIVO CRIPTO*\n• Régimen BTC: `{regimen_btc}`\n• Riesgo Dinámico: `{riesgo_actual*100:.2f}%%`\n\n'

  for nombre_grupo, lista_simbolos, temporalidad in portafolios:
    reporte_general += f'📂 *--- {nombre_grupo} ---*\n'
    for simbolo in lista_simbolos:
      if not es_mercado_spot_valido(exchange, simbolo):
        continue

      velas = descargar_velas_cerradas(exchange, simbolo, temporalidad, VELAS_ANALISIS)
      if not velas or len(velas) < 200:
        continue

      df = pd.DataFrame(
          velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen']
      )
      df['RSI'] = calcular_rsi(df['cierre'])
      df['ATR'] = calcular_atr(df)
      df['ADX'] = calcular_adx(df)
      df = calcular_cvd(df)
      df['Hurst'] = calcular_hurst_vectorizado(df['cierre'])
      slope, r2 = calcular_regresion_rolling(df['cierre'])
      df['Pendiente_Reg'] = slope
      df['R2_Tendencia'] = r2

      _, sa, sb = detectar_patrones_smc_historico(df)
      df['CRT_Valida'] = sa | sb

      df = df.dropna()
      if len(df) < 50:
        continue

      ultima = df.iloc[-1]
      precio = ultima['cierre']
      rsi = ultima['RSI']
      atr = ultima['ATR']
      adx = ultima['ADX']
      hurst = ultima['Hurst']
      slope = ultima['Pendiente_Reg']
      r2 = ultima['R2_Tendencia']
      crt = 1 if ultima['CRT_Valida'] else 0
      cvd = ultima['CVD']
      delta_vol = ultima['Delta_Volumen']

      prob_subida, prob_bajada = inferir_probabilidad_xgboost(df, simbolo)
      tiempo_hold, recomendacion_hold = estimar_pico_confluencia(df)

      # Registrar predicción para autoaprendizaje
      if prob_subida is not None:
        registrar_prediccion(
            simbolo,
            precio,
            rsi,
            atr,
            adx,
            ultima['volumen'],
            hurst,
            slope,
            r2,
            crt,
            cvd,
            delta_vol,
            prob_subida,
        )

      estado_prob = f'{prob_subida:.1f}%' if prob_subida is not None else 'Calculando...'
      reporte_general += f'• `{simbolo}` | Precio: `{precio}` | RSI: `{rsi:.1f}` | Prob. IA: `{estado_prob}` | {recomendacion_hold}\n'

    reporte_general += '\n'

  enviar_alerta_telegram(reporte_general)
  print('✅ Ciclo de criptomonedas ejecutado y reportado con éxito.')


if __name__ == '__main__':
  ejecutar_bot_cripto()
