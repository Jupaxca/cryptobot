import ccxt
import pandas as pd
import numpy as np
import os
import requests

CORE_ASSETS = ['BTC/USD', 'ETH/USD', 'SOL/USD']
TEMPORALIDAD = '1d'

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calcular_bollinger_bands(series, period=20, std_dev=2):
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def enviar_alerta_telegram(mensaje):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Faltan las credenciales de Telegram.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("📱 Alerta institucional definitiva enviada con éxito.")
        else:
            print(f"❌ Error al enviar Telegram: {response.text}")
    except Exception as e:
        print(f"❌ Excepción en Telegram: {e}")

def ejecutar_bot_maestro():
    exchange = ccxt.kraken()
    print("="*60)
    print(" ESCÁNER CUANTITATIVO MAESTRO 24/7 (CORE-SATELLITE + INTELIGENCIA DCA)")
    print("="*60)
    
    alertas_core = []
    alertas_satelite = []
    btc_saludable = True

    # --- ESCUDO MACRO: ANÁLISIS DE TENDENCIA DE BITCOIN ---
    print("\n🛡️ Verificando Escudo Macro (Tendencia de Bitcoin)...")
    try:
        velas_btc = exchange.fetch_ohlcv('BTC/USD', timeframe=TEMPORALIDAD, limit=60)
        if velas_btc and len(velas_btc) >= 50:
            df_btc = pd.DataFrame(velas_btc, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
            df_btc['SMA_50'] = df_btc['cierre'].rolling(window=50).mean()
            precio_btc = df_btc.iloc[-1]['cierre']
            sma_50_btc = df_btc.iloc[-1]['SMA_50']
            
            if precio_btc < sma_50_btc:
                btc_saludable = False
                print("  ⚠️ ADVERTENCIA: Bitcoin por debajo de su SMA 50. Bloqueo de Satélites.")
            else:
                print("  ✅ Bitcoin en tendencia macro saludable. Luz verde para satélites.")
    except Exception as e:
        print(f"⚠️ Error al verificar el escudo macro de BTC: {e}")

    # --- BLOQUE CONSERVADOR (95% - Core / Acumulación Inteligente) ---
    print("\n🛡️ Analizando Núcleo Conservador con Diagnóstico Táctico...")
    for simbolo in CORE_ASSETS:
        try:
            velas = exchange.fetch_ohlcv(simbolo, timeframe=TEMPORALIDAD, limit=50)
            if not velas or len(velas) < 35:
                continue
                
            df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
            df['RSI'] = calcular_rsi(df['cierre'], period=14)
            df['Media_30'] = df['cierre'].rolling(window=30).mean()
            
            precio = df.iloc[-1]['cierre']
            rsi = df.iloc[-1]['RSI']
            media_30 = df.iloc[-1]['Media_30']
            
            # LÓGICA DE INTERPRETACIÓN TÁCTICA:
            # 1. Si el precio está por debajo de la media de 30 días o RSI < 40 -> COMPRAR (Descuento)
            # 2. Si el RSI > 75 -> EVALUAR VENTA (Sobrecalentamiento)
            # 3. Si el precio está muy por arriba de la media y RSI normal -> ESPERAR (Precio Caro)
            
            if precio <= (media_30 * 0.96) or rsi < 40:
                accion = "🟢 *COMPRAR* (Activo en descuento por corrección)"
            elif rsi >= 75:
                accion = "🟡 *EVALUAR TOMA DE BENEFICIOS* (Sobrecalentamiento extremo)"
            elif precio > (media_30 * 1.05):
                accion = "🔴 *ESPERAR* (Precio por encima del promedio, mantén liquidez)"
            else:
                accion = "⚪ *ZONA NEUTRA* (Esperar mejor estructura)"

            alertas_core.append({
                'simbolo': simbolo,
                'precio': precio,
                'rsi': rsi,
                'media_30': media_30,
                'accion': accion
            })
        except Exception as e:
            print(f"⚠️ Error en Core {simbolo}: {e}")

    # --- BLOQUE DE ALTO RIESGO / REBOTE (5% - Satélite) ---
    if btc_saludable:
        print("\n🚀 Analizando Satélite de Corto Plazo (Filtros Avanzados)...")
        try:
            tickers = exchange.fetch_tickers()
            altcoins_candidatas = []
            for s, ticker in tickers.items():
                if '/USD' in s and s not in CORE_ASSETS and 'USDT' not in s and 'USDC' not in s:
                    volumen_usd = ticker.get('quoteVolume', 0)
                    if volumen_usd and volumen_usd > 1000000:
                        altcoins_candidatas.append((s, volumen_usd))
            
            altcoins_candidatas.sort(key=lambda x: x[1], reverse=True)
            top_altcoins = [item[0] for item in altcoins_candidatas[:5]]

            for simbolo in top_altcoins:
                velas = exchange.fetch_ohlcv(simbolo, timeframe=TEMPORALIDAD, limit=50)
                if not velas or len(velas) < 30:
                    continue
                df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
                df['RSI'] = calcular_rsi(df['cierre'], period=14)
                upper, middle, lower = calcular_bollinger_bands(df['cierre'])
                df['BB_Lower'] = lower
                df['Vol_Medio'] = df['volumen'].rolling(window=20).mean()

                precio = df.iloc[-1]['cierre']
                rsi = df.iloc[-1]['RSI']
                bb_lower = df.iloc[-1]['BB_Lower']
                volumen_actual = df.iloc[-1]['volumen']
                volumen_promedio = df.iloc[-1]['Vol_Medio']

                if rsi <= 33 and precio <= bb_lower * 1.01 and volumen_actual >= (volumen_promedio * 0.7):
                    tp = precio * 1.08  
                    sl = precio * 0.95  
                    alertas_satelite.append({
                        'simbolo': simbolo, 'precio': precio, 'rsi': rsi, 'tp': tp, 'sl': sl
                    })
        except Exception as e:
            print(f"⚠️ Error analizando satélites avanzados: {e}")
    else:
        print("\n🚫 Satélites bloqueados por el Escudo Macro de Bitcoin.")

    # --- CONSTRUCCIÓN DEL REPORTE FINAL ---
    if alertas_core or alertas_satelite:
        mensaje = "🚨 *REPORTE INSTITUCIONAL MAESTRO CON INTELIGENCIA TÁCTICA* 🚨\n\n"
        
        if alertas_core:
            mensaje += "🛡️ *BLOQUE CONSERVADOR (95% - Núcleo / DCA)*\n"
            for op in alertas_core:
                mensaje += (
                    f"• *{op['simbolo']}* | Precio: `${op['precio']:,.2f}` | RSI: `{op['rsi']:.1f}`\n"
                    f"  Acción Sugerida: {op['accion']}\n\n"
                )
        
        if alertas_satelite:
            mensaje += "🚀 *BLOQUE DE ALTO RIESGO (5% - Satélite)*\n"
            for op in alertas_satelite:
                mensaje += (
                    f"• *{op['simbolo']}* (Bollinger + Volumen + Escudo BTC OK)\n"
                    f"  Entrada: `${op['precio']:,.2f}` | RSI: `{op['rsi']:.1f}`\n"
                    f"  🎯 *Take Profit (+8%):* `${op['tp']:,.2f}`\n"
                    f"  🛑 *Stop Loss (-5%):* `${op['sl']:,.2f}`\n\n"
                )
                
        mensaje += "💡 _Sistema autónomo 24/7 con gestión de acumulación inteligente._"
        enviar_alerta_telegram(mensaje)
    else:
        print("\nℹ️ Monitoreo sin activaciones.")

if __name__ == '__main__':
    executar_bot_maestro()
