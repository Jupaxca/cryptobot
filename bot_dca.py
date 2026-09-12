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
            print("📱 Alerta de prueba enviada con éxito a Telegram.")
        else:
            print(f"❌ Error al enviar Telegram: {response.text}")
    except Exception as e:
        print(f"❌ Excepción en Telegram: {e}")

def ejecutar_bot_maestro():
    exchange = ccxt.kraken()
    print("="*60)
    print(" PRUEBA DE FUEGO: ESCÁNER MAESTRO (MODO TEST FORZADO)")
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
            print(f"  Bitcoin Precio: ${precio_btc:,.2f} | SMA 50: ${sma_50_btc:,.2f}")
    except Exception as e:
        print(f"⚠️ Error al verificar BTC: {e}")

    # --- BLOQUE CONSERVADOR (95% - Core) [MODIFICADO PARA PRUEBA] ---
    print("\n🛡️ Analizando Núcleo Conservador (Modo Test)...")
    for simbolo in CORE_ASSETS:
        try:
            velas = exchange.fetch_ohlcv(simbolo, timeframe=TEMPORALIDAD, limit=50)
            if not velas or len(velas) < 30:
                continue
            df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
            df['RSI'] = calcular_rsi(df['cierre'], period=14)
            
            precio = df.iloc[-1]['cierre']
            rsi = df.iloc[-1]['RSI']
            
            # FILTRO DE PRUEBA: Forzamos la alerta
            if rsi <= 100:
                alertas_core.append({'simbolo': simbolo, 'precio': precio, 'rsi': rsi})
        except Exception as e:
            print(f"⚠️ Error en Core {simbolo}: {e}")

    # --- BLOQUE DE ALTO RIESGO / REBOTE (5% - Satélite) [MODIFICADO PARA PRUEBA] ---
    if btc_saludable:
        print("\n🚀 Analizando Satélite de Corto Plazo (Modo Test)...")
        try:
            tickers = exchange.fetch_tickers()
            altcoins_candidatas = []
            for s, ticker in tickers.items():
                if '/USD' in s and s not in CORE_ASSETS and 'USDT' not in s and 'USDC' not in s:
                    volumen_usd = ticker.get('quoteVolume', 0)
                    if volumen_usd and volumen_usd > 1000000:
                        altcoins_candidatas.append((s, volumen_usd))
            
            altcoins_candidatas.sort(key=lambda x: x[1], reverse=True)
            top_altcoins = [item[0] for item in altcoins_candidatas[:2]] # Tomamos 2 para la prueba
            
            for simbolo in top_altcoins:
                velas = exchange.fetch_ohlcv(simbolo, timeframe=TEMPORALIDAD, limit=50)
                if not velas or len(velas) < 30:
                    continue
                df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
                df['RSI'] = calcular_rsi(df['cierre'], period=14)
                
                precio = df.iloc[-1]['cierre']
                rsi = df.iloc[-1]['RSI']
                
                # FILTRO DE PRUEBA: Forzamos la alerta para ver Take Profit y Stop Loss
                if rsi <= 90:
                    tp = precio * 1.08   # Take Profit +8%
                    sl = precio * 0.95   # Stop Loss -5%
                    alertas_satelite.append({
                        'simbolo': simbolo, 
                        'precio': precio, 
                        'rsi': rsi, 
                        'tp': tp, 
                        'sl': sl
                    })
        except Exception as e:
            print(f"⚠️ Error analizando satélites: {e}")

    # --- CONSTRUCCIÓN DEL REPORTE FINAL ---
    if alertas_core or alertas_satelite:
        mensaje = "🚨 *MENSAJE DE PRUEBA - REPORTE MAESTRO* 🚨\n\n"
        
        if alertas_core:
            mensaje += "🛡️ *BLOQUE CONSERVADOR (95% - Núcleo)*\n"
            for op in alertas_core:
                mensaje += f"• *{op['simbolo']}* | Precio: `${op['precio']:,.2f}` | RSI: `{op['rsi']:.1f}`\n"
            mensaje += "\n"
            
        if alertas_satelite:
            mensaje += "🚀 *BLOQUE DE ALTO RIESGO (5% - Satélite)*\n"
            for op in alertas_satelite:
                mensaje += (
                    f"• *{op['simbolo']}* (Modo Prueba Forzada)\n"
                    f"  Entrada: `${op['precio']:,.2f}` | RSI: `{op['rsi']:.1f}`\n"
                    f"  🎯 *Take Profit (+8%):* `${op['tp']:,.2f}`\n"
                    f"  🛑 *Stop Loss (-5%):* `${op['sl']:,.2f}`\n\n"
                )
        
        mensaje += "💡 _Esta es una notificación de prueba para verificar tu bot._"
        enviar_alerta_telegram(mensaje)
    else:
        print("\nℹ️ No se generaron alertas.")

if __name__ == '__main__':
    ejecutar_bot_maestro()
