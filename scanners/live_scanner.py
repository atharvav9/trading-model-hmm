import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
import argparse
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load Environment Variables
load_dotenv()

# =========================
# CONFIG
# =========================
TIMEFRAME = mt5.TIMEFRAME_M5
EMA_LENGTH = 200
ATR_MULT   = 1.2   
POLL_INTERVAL = 5 

# =========================
# INIT MT5
# =========================
def init_mt5():
    account = os.getenv("MT5_ACCOUNT")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER", "MetaQuotes-Demo")

    if not account or not password:
        print("❌ Error: Missing MT5 credentials in .env file.")
        return False

    if not mt5.initialize():
        print("❌ MT5 Init Failed", mt5.last_error())
        return False

    if not mt5.login(int(account), password, server):
        print("❌ Login Failed", mt5.last_error())
        mt5.shutdown()
        return False
    return True

def get_analysis(symbol):
    # Pulling 500 bars for calculation stability
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, 500)
    if rates is None or len(rates) < 2: 
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Identify the bar that just CLOSED (index -2)
    closed_bar = df.iloc[-2]
    
    # 1. EMA 200
    df['ema'] = df['close'].ewm(span=EMA_LENGTH, adjust=False).mean()
    
    # 2. ATR (14)
    df['h_l'] = df['high'] - df['low']
    df['atr'] = df['h_l'].rolling(14).mean()
    
    # 3. Typical Price VWAP (Daily Reset)
    df['date'] = df['time'].dt.date
    current_date = closed_bar['time'].date()
    today_df = df[df['date'] == current_date].copy()
    
    if not today_df.empty:
        # Typical Price formula: (H + L + C) / 3
        today_df['tp'] = (today_df['high'] + today_df['low'] + today_df['close']) / 3
        today_df['pv'] = today_df['tp'] * today_df['tick_volume']
        vwap_val = (today_df['pv'].cumsum() / today_df['tick_volume'].cumsum()).iloc[-1]
    else:
        vwap_val = (closed_bar['high'] + closed_bar['low'] + closed_bar['close']) / 3
    
    return {
        "price": closed_bar['close'],
        "vwap": vwap_val,
        "ema": df['ema'].iloc[-2],
        "atr": df['atr'].iloc[-2],
        "upper": df['ema'].iloc[-2] + (df['atr'].iloc[-2] * ATR_MULT),
        "lower": df['ema'].iloc[-2] - (df['atr'].iloc[-2] * ATR_MULT),
        "time": closed_bar['time']
    }

# =========================
# MONITORING LOOP
# =========================
def run_scanner(symbol):
    if not init_mt5():
        return

    last_bar_time = None
    try:
        print(f"🚀 Monitoring {symbol} | ATR Mult: {ATR_MULT}x | VWAP: Typical Price")
        
        while True:
            data = get_analysis(symbol)
            
            if data is None:
                time.sleep(POLL_INTERVAL)
                continue

            # Check if a new bar has actually closed
            if last_bar_time != data['time']:
                last_bar_time = data['time']
                
                price = data['price']
                vwap = data['vwap']
                upper = data['upper']
                lower = data['lower']

                # SIGNAL LOGIC
                signal = "WAITING"
                if price > vwap and price > upper:
                    signal = "BUY"
                elif price < vwap and price < lower:
                    signal = "SELL"

                # Terminal Output
                print(f"\nBAR CLOSED: {data['time']}")
                print(f"Price: {price:.5f} | VWAP: {vwap:.5f} | ATR: {data['atr']:.5f}")
                print(f"EMA Gate: {lower:.5f} <---> {upper:.5f}")
                
                if signal != "WAITING":
                    color = "\033[92m" if signal == "BUY" else "\033[91m"
                    print(f"{color}>>> {signal} SIGNAL CONFIRMED <<<\033[0m")
                    print("\a") # System Alert Beep
                else:
                    print("Status: Scanning...")
            
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopping Monitor...")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live MT5 Scanner for VWAP & EMA Breakouts.")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., USDJPY, GBPUSD)")
    args = parser.parse_args()
    
    run_scanner(args.symbol.upper())
