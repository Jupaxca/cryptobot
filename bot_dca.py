import ccxt
import pandas as pd
import numpy as np
import os
import requests

# ==========================================================================
# CONFIGURACIÓN DEL SUPER BOT 24/7 (ESTRATEGIA CORE-SATELLITE)
# ==========================================================================
# 1. BLOQUE CONSERVADOR (95%): Los gigantes de alta liquidez y refugio
CORE_ASSETS = ['BTC/USD', 'ETH/USD', 'SOL/USD']

# 2. BLOQUE DE ALTO RIESGO / POTENCIAL (5%): Altcoins dinámicas de alto volumen
# El bot filtrará automáticamente las de mayor movimiento en el mercado.
TEMPORALIDAD = '1d'

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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
            print("📱 Alerta Core-Satellite enviada a Telegram con éxito.")
        else:
            print(f"❌ Error al enviar Telegram: {response.text}")
    except Exception as e:
        print(f"❌ Excepción en Telegram: {e}")

def ejecutar_bot_inteligente():
    exchange = ccxt.kraken()
    print("="*60)
    print(" INICIANDO ESCÁNER 24/7 (ESTRATEGIA CORE-SATELLITE)")
    print("="*60)
    
    alertas_core = []
    alertas_satelite = []

    # --- ANÁLISIS DEL BLOQUE CONSERVADOR (95%) ---
    print("\n🛡️ Analizando Bloque Conservador (Core)...")
    for simbolo in CORE_ASSETS:
        try:
            velas = exchange.fetch_ohlcv(simbolo, timeframe=TEMPORALIDAD, limit=50)
            if not velas or len(velas) < 30:
                continue
            df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
            df['RSI'] = calcular_rsi(df['cierre'], period=14)
            
            precio = df.iloc[-1]['cierre']
            rsi = df.iloc[-1]['RSI']
            print(f"  [Core] {simbolo:<10} | Precio: ${precio:,.2f} | RSI: {rsi:.1f}")
            
            # Regla conservadora: RSI bajo para acumulación segura en el núcleo
            if rsi < 40:
                alertas_core.append({'simbolo': simbolo, 'precio': precio, 'rsi': rsi})
        except Exception as e:
            print(f"⚠️ Error en Core {simbolo}: {e}")

    # --- ANÁLISIS DEL BLOQUE DE ALTO RIESGO / SATÉLITE (5%) ---
    print("\n🚀 Analizando Bloque de Alto Riesgo (Satélite)...")
    try:
        tickers = exchange.fetch_tickers()
        # Filtrar pares contra USD con buen volumen diario en Kraken
        altcoins_candidatas = []
        for s, ticker in tickers.items():
            if '/USD' in s and s not in CORE_ASSETS and 'USDT' not in s and 'USDC' not in s:
                volumen_usd = ticker.get('quoteVolume', 0)
                if volumen_usd and volumen_usd > 1000000: # Al menos 1M USD de volumen diario
                    altcoins_candidatas.append((s, volumen_usd))
        
        # Ordenar por mayor volumen de negociación y tomar los primeros 5 más activos
        altcoins_candidatas.sort(key=lambda x: x[1], reverse=True)
        top_altcoins = [item[0] for item in altcoins_candidatas[:5]]
        
        for simbolo in top_altcoins:
            velas = exchange.fetch_ohlcv(simbolo, timeframe=TEMPORALIDAD, limit=50)
            if not velas or len(velas) < 30:
                continue
            df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
            df['RSI'] = calcular_rsi(df['cierre'], period=14)
            
            precio = df.iloc[-1]['cierre']
            rsi = df.iloc[-1]['RSI']
            print(f"  [Satélite] {simbolo:<10} | Precio: ${precio:,.2f} | RSI: {rsi:.1f}")
            
            # Regla agresiva de corto plazo: Sobreventa fuerte (RSI < 32) para buscar rebotes rápidos
            if rsi < 32:
                tp = precio * 1.08   # Take Profit +8%
                sl = precio * 0.95   # Stop Loss -5% (mayor margen por volatilidad)
                alertas_satelite.append({'simbolo': simbolo, 'precio': precio, 'rsi': rsi, 'tp': tp, 'sl': sl})
    except Exception as e:
        print(f"⚠️ Error analizando satélites: {e}")

    # --- CONSTRUCCIÓN DEL MENSAJE DE TELEGRAM ---
    if alertas_core or alertas_satelite:
        mensaje = "🚨 *REPORTE TÁCTICO DE MERCADO (24/7)* 🚨\n\n"
        
        if alertas_core:
            mensaje += "🛡️ *BLOQUE CONSERVADOR (95% - Núcleo)*\n"
            for op in alertas_core:
                mensaje += f"• *{op['simbolo']}* | Precio: `${op['precio']:,.2f}` | RSI: `{op['rsi']:.1f}` (Zona de Acumulación Segura)\n"
            mensaje += "\n"
            
        if alertas_satelite:
            mensaje += "🚀 *BLOQUE DE ALTO RIESGO / POTENCIAL (5% - Satélite)*\n"
            for op in alertas_satelite:
                mensaje += (
                    f"• *{op['simbolo']}*\n"
                    f"  Entrada: `${op['precio']:,.2f}` | RSI: `{op['rsi']:.1f}`\n"
                    f"  🎯 *Take Profit (+8%):* `${op['tp']:,.2f}`\n"
                    f"  🛑 *Stop Loss (-5%):* `${op['sl']:,.2f}`\n\n"
                )
        
        mensaje += "💡 _Ejecución autónoma horaria._"
        enviar_alerta_telegram(mensaje)
    else:
        print("\nℹ️ Monitoreo 24/7 exitoso. Ningún activo cumple criterios estrictos en este ciclo horario.")

if __name__ == '__main__':
    ejecutar_bot_inteligente()
