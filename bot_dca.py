import ccxt
import pandas as pd
import numpy as np
import itertools
import time

# ==========================================================================
# 1. CONFIGURACIÓN GLOBAL (Reversión a la Media - RSI + Filtro de Régimen)
# ==========================================================================
SIMBOLOS = ['UNI/USD', 'ETH/USD', 'LINK/USD', 'SOL/USD', 'BTC/USD']
TEMPORALIDAD = '4h'
CAPITAL_INICIAL = 370000
COMISION = 0.0025
SLIPPAGE = 0.0010
RIESGO_POR_TRADE = 0.01

# 4h con 3000 velas = ~500 días. Para cobertura comparable al análisis diario
# (~8 años) hace falta bastante más historial. 12000 velas de 4h = ~2000 días
# (~5.5 años). Ajusta según la paciencia con el rate limit de Kraken:
# ~12 llamadas de 1000 velas por activo, con pausa entre cada una.
MAX_VELAS = 12000

# Grid de parámetros a optimizar. Se mezclan tus valores originales con los
# valores "de manual" (documentados en la bibliografía clásica de análisis
# técnico) para que el walk-forward compita ambos conjuntos y elija por
# evidencia out-of-sample, no por autoridad del libro:
#   - RSI: Wilder (1978) usa 30/70 como estándar; 20/80 es la variante más
#     citada para mercados con tendencias fuertes (aplicable a cripto).
#   - ADX: Wilder también define 25 como "tendencia confirmada" y ~20 como
#     "sin tendencia clara" (régimen de rango).
#   - Multiplicador ATR del stop: Van Tharp documenta un rango de 1.5x-3.0x
#     como uso profesional estándar; se cubre el rango completo.
PARAMS_RSI_COMPRA = [20, 25, 30, 35]   # 20 y 30 = valores de manual; 25 y 35 = tus originales
PARAMS_ATR = [1.5, 2.0, 2.5, 3.0]      # rango completo documentado por Van Tharp
PARAMS_ADX_MAX = [20, 25, 30]          # 20 y 25 = valores de manual (Wilder); 30 = tu original

# Filtro de tendencia mayor. Se ofrecen dos variantes en el grid:
#   - 'ema200'       : precio por encima de la EMA de 200 (tu versión original)
#   - 'golden_cross'  : SMA_50 por encima de SMA_200 (convención estándar de
#                        análisis técnico, señal de régimen alcista de largo plazo)
PARAMS_FILTRO_TENDENCIA = ['ema200', 'golden_cross']
EMA_FILTRO_TENDENCIA = 200
SMA_CORTA_GOLDEN_CROSS = 50
SMA_LARGA_GOLDEN_CROSS = 200

N_FOLDS_WALK_FORWARD = 4
MIN_TRADES_CONFIABLE = 20
N_ITER_BOOTSTRAP = 2000


# ==========================================================================
# 2. DESCARGA DE DATOS
# ==========================================================================
def descargar_historial_completo(exchange, simbolo, temporalidad, max_velas=3000):
    tf_ms = exchange.parse_timeframe(temporalidad) * 1000
    ahora = exchange.milliseconds()
    since = ahora - max_velas * tf_ms

    todas_las_velas = []
    while True:
        try:
            velas = exchange.fetch_ohlcv(simbolo, timeframe=temporalidad, since=since, limit=1000)
            if not velas:
                break
            todas_las_velas += velas
            ultimo_ts = velas[-1][0]
            nuevo_since = ultimo_ts + tf_ms
            if nuevo_since <= since or nuevo_since >= ahora:
                break
            since = nuevo_since
            if len(todas_las_velas) >= max_velas:
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception:
            break

    if not todas_las_velas:
        return pd.DataFrame()

    df = pd.DataFrame(todas_las_velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    return df


# ==========================================================================
# 3. INDICADORES: RSI, ATR, ADX (filtro de régimen) y EMA de tendencia
# ==========================================================================
def calcular_indicadores(df):
    df = df.copy()

    # --- RSI ---
    delta = df['cierre'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- ATR (gestión de riesgo) ---
    tr0 = df['maximo'] - df['minimo']
    tr1 = (df['maximo'] - df['cierre'].shift(1)).abs()
    tr2 = (df['minimo'] - df['cierre'].shift(1)).abs()
    df['TR'] = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # --- ADX (filtro de régimen: bajo = mercado en rango, favorable para reversión) ---
    up_move = df['maximo'] - df['maximo'].shift(1)
    down_move = df['minimo'].shift(1) - df['minimo']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / df['ATR'])
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / df['ATR'])
    suma_di = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / suma_di
    dx = dx.replace([np.inf, -np.inf], np.nan)
    df['ADX'] = dx.rolling(14).mean()

    # --- EMA de tendencia mayor (filtro: evitar comprar dentro de un bear market) ---
    df['EMA_TENDENCIA'] = df['cierre'].ewm(span=EMA_FILTRO_TENDENCIA, adjust=False).mean()

    # --- Golden Cross: SMA corta vs SMA larga, filtro de tendencia alternativo ---
    df['SMA_CORTA'] = df['cierre'].rolling(window=SMA_CORTA_GOLDEN_CROSS).mean()
    df['SMA_LARGA'] = df['cierre'].rolling(window=SMA_LARGA_GOLDEN_CROSS).mean()

    return df.dropna().reset_index(drop=True)


# ==========================================================================
# 4. BACKTEST (con filtro de régimen ADX + filtro de tendencia EMA)
# ==========================================================================
def backtest(df, atr_mult, rsi_compra_thresh, adx_max_thresh, filtro_tendencia, capital_inicial, riesgo_por_trade):
    efectivo = capital_inicial
    en_posicion = False
    precio_entrada = stop_loss = cantidad = 0.0
    trades_pnl = []
    equity_curve = [capital_inicial]

    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        precio_cierre = row['cierre']

        if en_posicion:
            nuevo_stop = precio_cierre - (atr_mult * row['ATR'])
            if nuevo_stop > stop_loss:
                stop_loss = nuevo_stop

            salida = None
            if row['minimo'] <= stop_loss:
                salida = stop_loss * (1 - SLIPPAGE)
            elif row['RSI'] >= 50.0:
                salida = precio_cierre * (1 - SLIPPAGE)

            if salida is not None:
                ingreso_neto = (cantidad * salida) * (1 - COMISION)
                efectivo += ingreso_neto
                trades_pnl.append((salida / precio_entrada) - 1)
                en_posicion = False
                cantidad = 0.0
                equity_curve.append(efectivo)
                continue
            else:
                equity_curve.append(efectivo + cantidad * precio_cierre)
                continue
        else:
            senal_sobreventa = (prev['RSI'] >= rsi_compra_thresh) and (row['RSI'] < rsi_compra_thresh)
            regimen_de_rango = row['ADX'] < adx_max_thresh          # filtro: NO operar en tendencia fuerte

            if filtro_tendencia == 'golden_cross':
                tendencia_favorable = row['SMA_CORTA'] > row['SMA_LARGA']
            else:  # 'ema200'
                tendencia_favorable = precio_cierre > row['EMA_TENDENCIA']

            if senal_sobreventa and regimen_de_rango and tendencia_favorable:
                precio_entrada = precio_cierre * (1 + SLIPPAGE)
                stop_loss = precio_entrada - (atr_mult * row['ATR'])
                distancia_riesgo = precio_entrada - stop_loss

                if distancia_riesgo > 0:
                    riesgo_dinero = efectivo * riesgo_por_trade
                    cantidad_por_riesgo = riesgo_dinero / distancia_riesgo
                    cantidad_maxima = efectivo / precio_entrada
                    cantidad = min(cantidad_por_riesgo, cantidad_maxima)
                    costo = cantidad * precio_entrada
                    efectivo -= costo
                    en_posicion = True

            equity_curve.append(efectivo if not en_posicion else efectivo + cantidad * precio_cierre)

    if en_posicion:
        salida = df.iloc[-1]['cierre'] * (1 - SLIPPAGE)
        ingreso_neto = (cantidad * salida) * (1 - COMISION)
        efectivo += ingreso_neto
        trades_pnl.append((salida / precio_entrada) - 1)
        equity_curve[-1] = efectivo

    return efectivo, trades_pnl, equity_curve


# ==========================================================================
# 5. MÉTRICAS COMPLETAS (periodos_por_anio=2190 para velas de 4h: 365*6)
# ==========================================================================
def calcular_metricas(capital_inicial, capital_final, trades_pnl, equity_curve, periodos_por_anio=2190):
    equity = np.array(equity_curve, dtype=float)
    retornos_por_vela = np.diff(equity) / equity[:-1]
    retornos_por_vela = retornos_por_vela[np.isfinite(retornos_por_vela)]

    retorno_total = (capital_final / capital_inicial - 1) * 100

    pico = np.maximum.accumulate(equity)
    drawdown = (equity - pico) / pico
    max_dd = drawdown.min() * 100 if len(drawdown) else 0.0

    if retornos_por_vela.std() > 0:
        sharpe = (retornos_por_vela.mean() / retornos_por_vela.std()) * np.sqrt(periodos_por_anio)
    else:
        sharpe = 0.0

    bajistas = retornos_por_vela[retornos_por_vela < 0]
    if len(bajistas) > 0 and bajistas.std() > 0:
        sortino = (retornos_por_vela.mean() / bajistas.std()) * np.sqrt(periodos_por_anio)
    else:
        sortino = 0.0

    velas_totales = max(len(equity), 1)
    retorno_anualizado = ((capital_final / capital_inicial) ** (periodos_por_anio / velas_totales) - 1) * 100
    calmar = (retorno_anualizado / abs(max_dd)) if max_dd != 0 else 0.0

    ganancias = [p for p in trades_pnl if p > 0]
    perdidas = [p for p in trades_pnl if p <= 0]
    profit_factor = (sum(ganancias) / abs(sum(perdidas))) if perdidas and sum(perdidas) != 0 else (np.inf if ganancias else 0.0)
    win_rate = (len(ganancias) / len(trades_pnl) * 100) if trades_pnl else 0.0

    return {
        'Retorno (%)': retorno_total,
        'Retorno Anualizado (%)': retorno_anualizado,
        'Max Drawdown (%)': max_dd,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Profit Factor': profit_factor,
        'Win Rate (%)': win_rate,
        'Trades': len(trades_pnl),
    }


# ==========================================================================
# 6. GRID SEARCH (3 parámetros: ATR, RSI, ADX_max) + ESTABILIDAD GENERALIZADA
# ==========================================================================
def grid_search(df, combinaciones, capital_inicial, riesgo_por_trade):
    resultados = []
    for atr_mult, rsi_thresh, adx_max, filtro_tendencia in combinaciones:
        capital_final, trades_pnl, equity_curve = backtest(
            df, atr_mult, rsi_thresh, adx_max, filtro_tendencia, capital_inicial, riesgo_por_trade
        )
        metricas = calcular_metricas(capital_inicial, capital_final, trades_pnl, equity_curve)
        metricas['ATR Multiplier'] = atr_mult
        metricas['RSI Compra'] = rsi_thresh
        metricas['ADX Max'] = adx_max
        metricas['Filtro Tendencia'] = filtro_tendencia
        resultados.append(metricas)
    return pd.DataFrame(resultados)


def puntuar_estabilidad(df_resultados, params_dict, columna='Sharpe'):
    """
    Versión generalizada a N parámetros (antes solo soportaba 2D).
    Promedia la métrica de cada combinación con sus vecinos inmediatos
    en cada dimensión del grid. Un parámetro rodeado de buenos vecinos es
    robusto; un pico aislado suele ser sobreajuste.
    """
    col_names = list(params_dict.keys())
    tabla = df_resultados.set_index(col_names)[columna]

    estabilidad = []
    for idx in tabla.index:
        idx_vals = idx if isinstance(idx, tuple) else (idx,)
        posiciones = [params_dict[c].index(v) for c, v in zip(col_names, idx_vals)]
        rangos = [range(p - 1, p + 2) for p in posiciones]

        vecinos = []
        for combo in itertools.product(*rangos):
            if all(0 <= combo[i] < len(params_dict[col_names[i]]) for i in range(len(col_names))):
                key = tuple(params_dict[col_names[i]][combo[i]] for i in range(len(col_names)))
                if key in tabla.index:
                    vecinos.append(tabla.loc[key])
        estabilidad.append(np.mean(vecinos))

    df_res = df_resultados.copy()
    df_res[f'{columna} Estabilidad'] = estabilidad
    return df_res


# ==========================================================================
# 7. BENCHMARK: BUY & HOLD
# ==========================================================================
def buy_and_hold(df, capital_inicial):
    precio_inicio = df.iloc[0]['cierre']
    precio_fin = df.iloc[-1]['cierre']
    cantidad = (capital_inicial * (1 - COMISION)) / (precio_inicio * (1 + SLIPPAGE))
    capital_final = cantidad * precio_fin * (1 - SLIPPAGE) * (1 - COMISION)
    equity_curve = capital_inicial * (df['cierre'] / precio_inicio).values
    retorno_pct = (capital_final / capital_inicial - 1) * 100

    pico = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - pico) / pico
    max_dd = drawdown.min() * 100 if len(drawdown) else 0.0

    return retorno_pct, max_dd


# ==========================================================================
# 8. BOOTSTRAP: intervalo de confianza sobre los trades OOS
# ==========================================================================
def bootstrap_confianza(trades_pnl, n_iter=2000, seed=7):
    if len(trades_pnl) < 3:
        return None
    rng = np.random.default_rng(seed)
    trades = np.array(trades_pnl)
    retornos_simulados = []
    for _ in range(n_iter):
        muestra = rng.choice(trades, size=len(trades), replace=True)
        retorno_compuesto = np.prod(1 + muestra) - 1
        retornos_simulados.append(retorno_compuesto * 100)
    retornos_simulados = np.array(retornos_simulados)
    p5, p50, p95 = np.percentile(retornos_simulados, [5, 50, 95])
    return {'p5': p5, 'mediana': p50, 'p95': p95}


# ==========================================================================
# 9. WALK-FORWARD ANALYSIS
# ==========================================================================
def walk_forward(df, combinaciones, params_dict, n_folds, capital_inicial, riesgo_por_trade):
    n = len(df)
    tam_fold = n // (n_folds + 1)
    resultados_oos = []
    trades_agregados = []

    for fold in range(1, n_folds + 1):
        fin_train = tam_fold * fold
        fin_test = min(tam_fold * (fold + 1), n)
        if fin_test <= fin_train:
            break

        df_train = df.iloc[:fin_train].reset_index(drop=True)
        df_test = df.iloc[fin_train:fin_test].reset_index(drop=True)
        if len(df_train) < 60 or len(df_test) < 10:
            continue

        res_train = grid_search(df_train, combinaciones, capital_inicial, riesgo_por_trade)
        res_train = puntuar_estabilidad(res_train, params_dict, columna='Sharpe')
        mejor = res_train.sort_values('Sharpe Estabilidad', ascending=False).iloc[0]
        atr_elegido = mejor['ATR Multiplier']
        rsi_elegido = mejor['RSI Compra']
        adx_elegido = mejor['ADX Max']
        filtro_elegido = mejor['Filtro Tendencia']
        sharpe_in_sample = mejor['Sharpe']

        capital_final, trades_pnl, equity_curve = backtest(
            df_test, atr_elegido, rsi_elegido, adx_elegido, filtro_elegido, capital_inicial, riesgo_por_trade
        )
        metricas_oos = calcular_metricas(capital_inicial, capital_final, trades_pnl, equity_curve)

        bh_retorno, bh_dd = buy_and_hold(df_test, capital_inicial)
        ratio_overfitting = (metricas_oos['Sharpe'] / sharpe_in_sample) if sharpe_in_sample != 0 else np.nan

        metricas_oos['Fold'] = fold
        metricas_oos['ATR'] = atr_elegido
        metricas_oos['RSI'] = rsi_elegido
        metricas_oos['ADX Max'] = adx_elegido
        metricas_oos['Filtro Tendencia'] = filtro_elegido
        metricas_oos['Sharpe In-Sample'] = sharpe_in_sample
        metricas_oos['Ratio OOS/IS'] = ratio_overfitting
        metricas_oos['Buy&Hold (%)'] = bh_retorno
        resultados_oos.append(metricas_oos)
        trades_agregados.extend(trades_pnl)

    return pd.DataFrame(resultados_oos), trades_agregados


# ==========================================================================
# 10. EJECUCIÓN GENERAL
# ==========================================================================
if __name__ == '__main__':
    exchange = ccxt.kraken()
    combinaciones = list(itertools.product(PARAMS_ATR, PARAMS_RSI_COMPRA, PARAMS_ADX_MAX, PARAMS_FILTRO_TENDENCIA))
    params_dict = {
        'ATR Multiplier': PARAMS_ATR,
        'RSI Compra': PARAMS_RSI_COMPRA,
        'ADX Max': PARAMS_ADX_MAX,
        'Filtro Tendencia': PARAMS_FILTRO_TENDENCIA,
    }
    print(f"Grid total: {len(combinaciones)} combinaciones "
          f"(mezcla de valores propios y valores de manual, elegidos por evidencia OOS)\n")
    resultados_globales = []
    filas_grid_export = []
    filas_folds_export = []

    print("="*95)
    print(" ESCÁNER (REVERSIÓN A LA MEDIA RSI + FILTRO DE RÉGIMEN ADX + FILTRO EMA200)")
    print("="*95)

    for simbolo in SIMBOLOS:
        print(f"\n{'#'*95}")
        print(f" ACTIVO: {simbolo}")
        print(f"{'#'*95}")

        df_raw = descargar_historial_completo(exchange, simbolo, TEMPORALIDAD, MAX_VELAS)
        if df_raw.empty or len(df_raw) < 250:
            print(f"⚠️ Historial insuficiente para {simbolo}. Saltando...")
            continue

        df_base = calcular_indicadores(df_raw)
        dias_cubiertos = len(df_base) * 4 / 24
        print(f"Velas utilizables: {len(df_base)}  (~{dias_cubiertos:.0f} días de historial)")

        print(f"\n--- SECCIÓN A: Grid search in-sample ({len(combinaciones)} combinaciones) ---")
        res_completo = grid_search(df_base, combinaciones, CAPITAL_INICIAL, RIESGO_POR_TRADE)
        res_completo = puntuar_estabilidad(res_completo, params_dict, columna='Sharpe')
        res_completo = res_completo.sort_values('Sharpe Estabilidad', ascending=False).reset_index(drop=True)

        cols_a = ['ATR Multiplier', 'RSI Compra', 'ADX Max', 'Filtro Tendencia', 'Retorno (%)',
                  'Retorno Anualizado (%)', 'Max Drawdown (%)', 'Sharpe', 'Sharpe Estabilidad', 'Sortino', 'Calmar',
                  'Profit Factor', 'Win Rate (%)', 'Trades']
        print(res_completo[cols_a].head(5).to_string(index=False))

        for _, r in res_completo.iterrows():
            fila = r.to_dict()
            fila['Activo'] = simbolo
            filas_grid_export.append(fila)

        bh_total_retorno, bh_total_dd = buy_and_hold(df_base, CAPITAL_INICIAL)
        print(f"\nBuy & Hold sobre todo el historial: {bh_total_retorno:+.2f}% | Max DD: {bh_total_dd:.2f}%")

        print(f"\n--- SECCIÓN B: Walk-forward ({N_FOLDS_WALK_FORWARD} folds, out-of-sample real) ---")
        res_wf, trades_oos_agregados = walk_forward(
            df_base, combinaciones, params_dict,
            N_FOLDS_WALK_FORWARD, CAPITAL_INICIAL, RIESGO_POR_TRADE
        )

        if res_wf.empty:
            print("No hay suficientes datos para correr walk-forward en este activo.")
            continue

        cols_wf = ['Fold', 'ATR', 'RSI', 'ADX Max', 'Filtro Tendencia', 'Retorno (%)', 'Buy&Hold (%)',
                   'Max Drawdown (%)', 'Sharpe', 'Sharpe In-Sample', 'Ratio OOS/IS', 'Sortino', 'Calmar',
                   'Profit Factor', 'Win Rate (%)', 'Trades']
        print(res_wf[cols_wf].to_string(index=False))

        for _, r in res_wf.iterrows():
            fila = r.to_dict()
            fila['Activo'] = simbolo
            filas_folds_export.append(fila)

        retorno_oos_compuesto = np.prod(1 + res_wf['Retorno (%)'] / 100) - 1
        retorno_bh_compuesto = np.prod(1 + res_wf['Buy&Hold (%)'] / 100) - 1
        sharpe_promedio = res_wf['Sharpe'].mean()
        drawdown_promedio = res_wf['Max Drawdown (%)'].mean()
        trades_totales = int(res_wf['Trades'].sum())
        win_rate_promedio = res_wf['Win Rate (%)'].mean()
        ratio_overfitting_promedio = res_wf['Ratio OOS/IS'].replace([np.inf, -np.inf], np.nan).mean()

        print(f"\nRetorno OOS compuesto (estrategia RSI): {retorno_oos_compuesto*100:+.2f}%")
        print(f"Retorno OOS compuesto (Buy & Hold):     {retorno_bh_compuesto*100:+.2f}%")
        print(f"Diferencia vs Buy & Hold:               {(retorno_oos_compuesto - retorno_bh_compuesto)*100:+.2f} pts")
        print(f"Max Drawdown promedio OOS: {drawdown_promedio:.2f}%")
        print(f"Sharpe promedio OOS: {sharpe_promedio:.2f}")
        print(f"Ratio Sharpe OOS/In-Sample promedio: {ratio_overfitting_promedio:.2f}  "
              f"(cercano a 1 = robusto; cercano a 0 o negativo = probable sobreajuste)")

        if trades_totales < MIN_TRADES_CONFIABLE:
            print(f"⚠️  Solo {trades_totales} trades OOS — muestra INSUFICIENTE "
                  f"(mínimo recomendado: {MIN_TRADES_CONFIABLE}).")
        else:
            print(f"✅ Muestra suficiente ({trades_totales} trades OOS).")

        boot = bootstrap_confianza(trades_oos_agregados, N_ITER_BOOTSTRAP)
        if boot:
            print(f"Intervalo de confianza (bootstrap, 90%): [{boot['p5']:+.2f}%, {boot['p95']:+.2f}%] "
                  f"| mediana: {boot['mediana']:+.2f}%")
            if boot['p5'] < 0 < boot['p95']:
                print("   -> El intervalo incluye 0%: no se puede afirmar con confianza que haya edge real.")
        else:
            print("Muestra demasiado chica para calcular un intervalo de confianza confiable.")

        resultados_globales.append({
            'Activo': simbolo,
            'Retorno OOS Compuesto (%)': retorno_oos_compuesto * 100,
            'Buy&Hold OOS Compuesto (%)': retorno_bh_compuesto * 100,
            'Diferencia vs B&H (pts)': (retorno_oos_compuesto - retorno_bh_compuesto) * 100,
            'Sharpe OOS Promedio': sharpe_promedio,
            'Ratio OOS/IS Promedio': ratio_overfitting_promedio,
            'Max Drawdown Promedio (%)': drawdown_promedio,
            'Win Rate Promedio (%)': win_rate_promedio,
            'Total Trades OOS': trades_totales,
            'Muestra Suficiente': trades_totales >= MIN_TRADES_CONFIABLE,
        })

    if resultados_globales:
        df_final = pd.DataFrame(resultados_globales).sort_values(
            by='Retorno OOS Compuesto (%)', ascending=False).reset_index(drop=True)

        print("\n" + "="*105)
        print(" RANKING GLOBAL DEFINITIVO (RSI + Filtro ADX + Filtro EMA200 — OOS vs Buy & Hold)")
        print("="*105)
        print(df_final.to_string(index=False))
        print("="*105)

        df_final.to_csv('ranking_global_rsi_v2.csv', index=False)
        pd.DataFrame(filas_grid_export).to_csv('grid_in_sample_rsi_v2.csv', index=False)
        pd.DataFrame(filas_folds_export).to_csv('walk_forward_rsi_v2.csv', index=False)
        print("\nArchivos exportados: ranking_global_rsi_v2.csv, grid_in_sample_rsi_v2.csv, walk_forward_rsi_v2.csv")
    else:
        print("No se pudieron generar resultados globales.")
