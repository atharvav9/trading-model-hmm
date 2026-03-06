import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime
import warnings
import os
import argparse
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

# =========================
# CONFIGURATION
# =========================
TIMEFRAME = mt5.TIMEFRAME_M5
BARS      = 300 # Increased for EMA/ATR stability
EMA_LEN   = 50
ATR_MULT  = 1.5  
VOL_CONT  = 0.7  

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

def get_data(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_indicators(df):
    df = df.copy()
    df['date'] = df['time'].dt.date
    
    # Typical Price for VWAP
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['tick_volume']
    
    # FIXED VWAP Logic
    df['vwap'] = df.groupby('date', group_keys=False).apply(
        lambda x: x['pv'].cumsum() / x['tick_volume'].cumsum()
    ).reset_index(level=0, drop=True)
    
    # Indicators
    df['ema'] = df['close'].ewm(span=EMA_LEN, adjust=False).mean()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['vol_avg'] = df['tick_volume'].rolling(20).mean()
    
    return df.dropna()

def check_logic(df):
    # We look at the bar that JUST CLOSED to avoid repainting
    curr = df.iloc[-1]   
    prev = df.iloc[-2]   
    p_imp = df.iloc[-3]  

    # 1. IMPULSE CHECK
    dist = abs(p_imp['close'] - p_imp['ema']) / p_imp['atr'] if p_imp['atr'] != 0 else 0
    is_impulse = dist > ATR_MULT and p_imp['tick_volume'] > p_imp['vol_avg']
    
    # 2. PULLBACK CHECK
    vol_ratio = prev['tick_volume'] / p_imp['tick_volume'] if p_imp['tick_volume'] != 0 else 0
    is_healthy_pb = vol_ratio < VOL_CONT
    
    # 3. TREND & TRIGGER
    signal = "NONE"
    if is_impulse and is_healthy_pb:
        # LONG
        if curr['close'] > curr['vwap'] and curr['ema'] > p_imp['ema']:
            if curr['close'] > prev['high']:
                signal = "🚀 STRONG BUY: Impulse + Low-Vol Pullback + Breakout"
        # SHORT
        elif curr['close'] < curr['vwap'] and curr['ema'] < p_imp['ema']:
            if curr['close'] < prev['low']:
                signal = "📉 STRONG SELL: Impulse + Low-Vol Pullback + Breakout"
                
    return signal, vol_ratio, dist

# =========================
# MAIN EXECUTION
# =========================
def run(symbol):
    if not init_mt5():
        return

    print(f"📡 Monitoring {symbol} | Impulse: {ATR_MULT}x ATR | PB Vol: <{VOL_CONT*100}%")
    last_time = None

    try:
        while True:
            df_raw = get_data(symbol, TIMEFRAME, BARS)
            if df_raw.empty: continue
            
            try:
                df = calculate_indicators(df_raw)
            except Exception as e:
                time.sleep(1)
                continue

            curr_time = df.iloc[-1]['time']
            
            if last_time != curr_time:
                last_time = curr_time
                sig, v_ratio, i_dist = check_logic(df)
                
                print(f"\n[{curr_time}] {symbol} Scan:")
                print(f"   Impulse Strength: {i_dist:.2f} ATR")
                print(f"   Pullback Vol %:   {v_ratio*100:.1f}%")
                
                if sig != "NONE":
                    print(f"   >>> {sig} <<<")
                    print("\a") # Alert beep
                else:
                    print("   Status: Searching for Setup...")
            
            time.sleep(5) 

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Impulse Monitor for MT5.")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol (e.g., XAUUSD)")
    args = parser.parse_args()
    
    run(args.symbol.upper())
