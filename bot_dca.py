import ccxt
import pandas as pd
import numpy as np
import os
import requests

# ==========================================================================
# CONFIGURACIÓN DEL BOT DE SWING TRADING (Corto Plazo: 5 a 15 días)
# ==========================================================================
# Canasta optimizada con alta liquidez y volatilidad para rebotes rápidos
SIMBOLOS = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'LINK/USD']
TEMPORALIDAD = '1d'  # Velas diarias para señales más limpias

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def calcular_rsi(series, period=14):
    """Calcula el indicador técnico RSI para detectar sobreventa"""
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
            print("📱 Alerta de Swing Trading enviada a Telegram con éxito.")
        else:
            print(f"❌ Error al enviar Telegram: {response.text}")
    except Exception as e:
        print(f"❌ Excepción en Telegram: {e}")

def ejecutar_escaner_swing():
    exchange = ccxt.kraken()
    print("="*60)
    print(" INICIANDO ESCÁNER DE SWING TRADING (Corto Plazo + Gestión de Riesgo)")
    print("="*60)
    
    oportunidades_detectadas = []

    for simbolo in SIMBOLOS:
        try:
            velas = exchange.fetch_ohlcv(simbolo, timeframe=TEMPORALIDAD, limit=100)
            if not velas or len(velas) < 30:
                continue
                
            df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
            df['RSI'] = calcular_rsi(df['cierre'], period=14)
            
            precio_actual = df.iloc[-1]['cierre']
            rsi_actual = df.iloc[-1]['RSI']
            
            print(f"Activo: {simbolo:<10} | Precio: ${precio_actual:,.2f} | RSI(14): {rsi_actual:.1f}")
            
            # REGLA DE SWING TRADING: RSI menor a 35 (Zona de sobreventa profunda / rebote inminente)
            if rsi_actual < 35:
                # Definir objetivos de gestión de riesgo automáticos
                take_profit = precio_actual * 1.07   # Meta: +7% de ganancia
                stop_loss = precio_actual * 0.965    # Protección: -3.5% de pérdida máxima
                
                oportunidades_detectadas.append({
                    'simbolo': simbolo,
                    'precio': precio_actual,
                    'rsi': rsi_actual,
                    'tp': take_profit,
                    'sl': stop_loss
                })
        except Exception as e:
            print(f"⚠️ Error procesando {simbolo}: {e}")

    # Si hay oportunidades, armar el reporte detallado con TP y SL
    if oportunidades_detectadas:
        mensaje = "🚨 *¡SEÑAL DE SWING TRADING DETECTADA!* 🚨\n\n"
        for op in oportunidades_detectadas:
            mensaje += (
                f"• *{op['simbolo']}*\n"
                f"  Precio de Entrada: `${op['precio']:,.2f}`\n"
                f"  RSI (14): `{op['rsi']:.1f}` (Zona de Rebote)\n"
                f"  🎯 *Take Profit (+7%):* `${op['tp']:,.2f}`\n"
                f"  🛑 *Stop Loss (-3.5%):* `${op['sl']:,.2f}`\n\n"
            )
        mensaje += "💡 _Operativa de corto plazo (5-15 días). Respeta siempre tu Stop-Loss._"
        enviar_alerta_telegram(mensaje)
    else:
        print("\nℹ️ Ningún activo está en zona de sobreventa estricta hoy. Esperando el momento ideal.")

if __name__ == '__main__':
    ejecutar_escaner_swing()
