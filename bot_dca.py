import ccxt
import pandas as pd
import numpy as np
import os
import requests

# ==========================================================================
# CONFIGURACIÓN DEL BOT DE ACUMULACIÓN (DCA INTELIGENTE) + TELEGRAM
# ==========================================================================
SIMBOLOS = ['BTC/USD', 'ETH/USD', 'UNI/USD', 'LINK/USD', 'SOL/USD', 'AVAX/USD']
TEMPORALIDAD = '1d'

# Leer credenciales de Telegram desde las variables de entorno de GitHub
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def enviar_alerta_telegram(mensaje):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Faltan las credenciales de Telegram en el entorno.")
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
            print("📱 Alerta de Telegram enviada con éxito.")
        else:
            print(f"❌ Error al enviar Telegram: {response.text}")
    except Exception as e:
        print(f"❌ Excepción en Telegram: {e}")

def ejecutar_escaner():
    exchange = ccxt.kraken()
    print("="*60)
    print(" INICIANDO ESCÁNER DIARIO DE ACUMULACIÓN (DCA) - TELEGRAM")
    print("="*60)
    
    oportunidades_detectadas = []

    for simbolo in SIMBOLOS:
        try:
            velas = exchange.fetch_ohlcv(simbolo, timeframe=TEMPORALIDAD, limit=100)
            if not velas or len(velas) < 35:
                continue
                
            df = pd.DataFrame(velas, columns=['timestamp', 'apertura', 'maximo', 'minimo', 'cierre', 'volumen'])
            df['Media_30'] = df['cierre'].rolling(window=30).mean()
            
            precio_actual = df.iloc[-1]['cierre']
            media_30 = df.iloc[-1]['Media_30']
            diferencia_pct = ((precio_actual - media_30) / media_30) * 100
            
            print(f"Activo: {simbolo:<10} | Precio: ${precio_actual:,.2f} | Distancia a Media: {diferencia_pct:+.2f}%")
            
            # REGLA DE COMPRA: Si el precio está al menos 5% por debajo de su media de 30 días
            if precio_actual <= (media_30 * 0.95):
                oportunidades_detectadas.append({
                    'simbolo': simbolo,
                    'precio': precio_actual,
                    'media': media_30,
                    'descuento': diferencia_pct
                })
        except Exception as e:
            print(f"⚠️ Error procesando {simbolo}: {e}")

    # Si hay oportunidades, armar el reporte y enviarlo por Telegram
    if oportunidades_detectadas:
        mensaje = "🚨 *¡OPORTUNIDADES DE COMPRA DCA DETECTADAS!* 🚨\n\n"
        for op in oportunidades_detectadas:
            mensaje += (
                f"• *{op['simbolo']}*\n"
                f"  Precio: `${op['precio']:,.2f}`\n"
                f"  Descuento: `{op['descuento']:.2f}%` (Bajo su media de 30d)\n\n"
            )
        mensaje += "💡 _Es momento de desplegar tu fraccionamiento de capital programado._"
        enviar_alerta_telegram(mensaje)
    else:
        print("\nℹ️ Ningún activo cumple la regla de descuento hoy. Manteniendo liquidez en efectivo.")

if __name__ == '__main__':
    ejecutar_escaner()
