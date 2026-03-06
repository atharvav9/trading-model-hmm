import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import warnings

warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
# Now looking for 3-minute files (e.g., 'xauusd_3m.csv')
ASSETS = ["xauusd", "eurusd", "gbpusd", "audusd", "nzdusd", "xagusd", "wti"]

# =========================
# 1. INTRADAY METRICS
# =========================
def calculate_metrics(df):
    # Typical Price
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Identify Volume Column
    vol_col = 'tickvol' if 'tickvol' in df.columns else 'vol'
    if vol_col not in df.columns:
        df[vol_col] = 1 # Fallback to avoid div/0
        
    df['pv'] = df['tp'] * df[vol_col]
    
    # ⚡ CRITICAL: SESSION VWAP (Resets Daily)
    # We group by the Date to ensure VWAP starts fresh at 00:00 each day
    # This creates the standard "Session VWAP" used by intraday traders
    df['date_only'] = df['date'].dt.date
    df['cum_pv'] = df.groupby('date_only')['pv'].cumsum()
    df['cum_vol'] = df.groupby('date_only')[vol_col].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_vol']
    
    # 3SD BANDS (Rolling 20 bars = 1 Hour context on 3m timeframe)
    # capturing short-term volatility spikes
    df['rolling_std'] = df['tp'].rolling(window=20).std()
    df['upper_3sd'] = df['vwap'] + (df['rolling_std'] * 3)
    df['lower_3sd'] = df['vwap'] - (df['rolling_std'] * 3)
    
    # ML FEATURES (Adjusted for 3m speed)
    
    # 1. RSI (14 bars = 42 mins)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 2. Volume Spike (Current vs Last 1 Hour Avg)
    df['vol_spike'] = df[vol_col] / df[vol_col].rolling(20).mean()
    
    # 3. ATR (14 bars = 42 mins)
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    return df.dropna()

# =========================
# 2. AUDIT ENGINE
# =========================
def run_ml_audit(asset):
    filename = f"{asset}_3m.csv"
    
    if not os.path.exists(filename):
        print(f"\n❌ Error: '{filename}' not found.")
        print("   (Ensure you have 3-minute data exported as tab-separated CSV)")
        return

    try:
        # Load Data (Handling MT5 Tab-Separated format)
        df = pd.read_csv(filename, sep='\t')
        df.columns = [c.replace('<', '').replace('>', '').lower() for c in df.columns]
        
        # Merge Date+Time for accurate Intraday parsing
        if 'time' in df.columns:
            df['date'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        else:
            df['date'] = pd.to_datetime(df['date'])
            
    except Exception as e:
        print(f"❌ Read Error: {e}")
        return
    
    # Calc Metrics
    df = calculate_metrics(df)
    
    # Identify Hits (3SD Touch)
    hits = []
    # Check 3 bars ahead (9 minutes) for scalp result
    for i in range(len(df) - 3):
        row = df.iloc[i]
        future_close = df.iloc[i+3]['close'] 
        
        # Upper Band Hit (Short Setup)
        if row['high'] >= row['upper_3sd']:
            # Success if price drops in 9 mins
            reversed = 1 if future_close < row['close'] else 0
            hits.append([row['rsi'], row['vol_spike'], row['atr'], reversed])
            
        # Lower Band Hit (Long Setup)
        elif row['low'] <= row['lower_3sd']:
            # Success if price rises in 9 mins
            reversed = 1 if future_close > row['close'] else 0
            hits.append([row['rsi'], row['vol_spike'], row['atr'], reversed])
            
    if not hits:
        print(f"\n⚠️ {asset.upper()}: Zero 3SD hits found in 3m data.")
        return

    # Results
    hits_df = pd.DataFrame(hits, columns=['RSI', 'Vol_Spike', 'ATR', 'Reversed'])
    total = len(hits_df)
    rev_count = hits_df['Reversed'].sum()
    rev_rate = (rev_count / total) * 100
    
    print("\n" + "⚡"*30)
    print(f"📊 INTRADAY REPORT: {asset.upper()} (3-Min)")
    print("⚡"*30)
    print(f"Total 3SD Hits:      {total}")
    print(f"Scalp Wins (9min):   {rev_count}")
    print(f"Win Rate:            {rev_rate:.2f}%")
    
    if total < 10:
        print("⚠️ Not enough hits for ML analysis.")
        return

    # ML Training
    X = hits_df[['RSI', 'Vol_Spike', 'ATR']]
    y = hits_df['Reversed']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Feature Importance
    imps = model.feature_importances_
    feats = X.columns
    sorted_idx = np.argsort(imps)[::-1]
    
    print("\n🧠 ML FACTOR ANALYSIS (Intraday Drivers)")
    print("-" * 40)
    for idx in sorted_idx:
        print(f"   > {feats[idx]}: {imps[idx]:.4f}")
    
    top = feats[sorted_idx[0]]
    print("-" * 40)
    if top == "RSI":
        print("💡 INSIGHT: Scalps work best on EXTREME MOMENTUM (High RSI).")
    elif top == "Vol_Spike":
        print("💡 INSIGHT: Scalps work best on VOLUME CLIMAX (Panic/Euphoria).")
    elif top == "ATR":
        print("💡 INSIGHT: Scalps work best on SUDDEN EXPANSION (Large Candles).")
    print("="*60)

# =========================
# 3. MAIN LOOP
# =========================
if __name__ == "__main__":
    while True:
        print("\n--- 3M INTRADAY VWAP AUDITOR ---")
        for i, asset in enumerate(ASSETS):
            print(f"{i+1}. {asset.upper()}")
        print("8. EXIT")
        
        choice = input("\nSelect Asset (1-8): ")
        
        if choice == '8':
            print("✌️ Later.")
            break
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(ASSETS):
                run_ml_audit(ASSETS[idx])
            else:
                print("❌ Invalid number.")
        except ValueError:
            print("❌ Enter a number.")