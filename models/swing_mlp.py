import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import os

# --- 1. DATA LOADER ---
def load_and_clean(file_path):
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path)
    df.columns = [c.lower() for c in df.columns]
    t_col = 'time' if 'time' in df.columns else 'date'
    df[t_col] = pd.to_datetime(df[t_col])
    df.set_index(t_col, inplace=True)
    df.sort_index(inplace=True)
    return df

# --- 2. INSTITUTIONAL FEATURE ENGINEERING ---
def add_v10_features(df):
    df = df.copy()
    
    # --- NEW: REGIME FILTER (The Bouncer) ---
    df['sma_200'] = df['close'].rolling(200).mean()
    
    # 1. EMA SLOPE
    for p in [13, 34, 55]:
        ema = df['close'].ewm(span=p).mean()
        df[f'ema_{p}_slope'] = (ema.diff(3) / ema) * 100
        
    # 2. STOCHASTICS
    low_14, high_14 = df['low'].rolling(14).min(), df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # 3. VOLUME FORCE
    df['vol_force'] = (df['close'].diff() * df['volume']) / df['volume'].rolling(20).mean()
    
    # 4. VOLATILITY
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    # TARGET
    df['target'] = (df['close'].shift(-4) > df['close']).astype(int)
    
    return df.dropna()

# --- 3. THE AUDITOR: PERFORMANCE REPORTING ---
def print_descriptive_report(equity, trade_log):
    returns = equity.pct_change().dropna()
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6) # Adjusted for 4H bars
    dd = (equity - equity.cummax()) / equity.cummax()
    
    wins = [t for t in trade_log if t > 0]
    win_rate = (len(wins) / len(trade_log)) * 100 if trade_log else 0
    start_bal = equity.iloc[0]
    lowest_point = equity.min()
    
    # If we ever went below start, calc the % drop
    if lowest_point < start_bal:
        abs_dd = (lowest_point - start_bal) / start_bal * 100
    else:
        abs_dd = 0.0
    
    print("\n" + "═"*60)
    print(f"{'V10 INSTITUTIONAL BACKTEST REPORT':^60}")
    print("═"*60)
    print("WTI")
    print(f"Total Return:         {total_ret:>15.2f}%")
    print(f"Max Drawdown:         {dd.min()*100:>15.2f}%")
    print(f"Sharpe Ratio:         {sharpe:>15.2f}")
    print(f"Total Trades:         {len(trade_log):>15}")
    print(f"Win Rate:             {win_rate:>15.2f}%")
    print(f"Profit Factor:        {abs(sum(wins)/sum([t for t in trade_log if t<0])) if len(trade_log)>len(wins) else 0:.2f}")
    print(f"Final Balance:        ${equity.iloc[-1]:>14,.2f}")
    print("═"*60)
    print(f"Drop from Start:        {abs_dd:>15.2f}%")

# --- 4. BACKTEST CORE ---
def run_v10_backtest(df):
    balance = 5000.0
    portfolio = [balance]
    trade_pnl_log = []
    active_pos = 0 
    entry_price, sl, tp = 0, 0, 0
    
    # Grab the numpy arrays for speed
    close = df['close'].values
    signals = df['signal'].values
    atr = df['atr'].values
    sma_200 = df['sma_200'].values  # <--- New Array
    
    for i in range(1, len(df)):
        # MANAGING OPEN TRADES
        if active_pos == 1:
            if close[i] >= tp or close[i] <= sl:
                # Same risk logic
                risk_amt = balance * 0.02
                dist_to_sl = entry_price - sl
                # Safety: avoid div by zero if SL is somehow on entry
                if dist_to_sl == 0: dist_to_sl = 0.0001 
                
                pos_size = risk_amt / dist_to_sl
                pnl = (close[i] - entry_price) * pos_size
                
                balance += pnl
                trade_pnl_log.append(pnl)
                active_pos = 0

        # LOOKING FOR ENTRIES
        # Rule: Signal is ON AND Price is ABOVE the 200 SMA
        elif signals[i] == 1 and close[i] > sma_200[i]: 
            active_pos = 1
            entry_price = close[i]
            sl = entry_price - (2.0 * atr[i])
            tp = entry_price + (4.0 * atr[i])
        
        portfolio.append(balance)
        
    return pd.Series(portfolio, index=df.index), trade_pnl_log


if __name__ == "__main__":
    # 1. Prepare Data
    p23, p24, p25 = "WTI_4H_2022.csv", "WTI_4H_2023.csv", "WTI_4H_2024.csv"
    train_df = pd.concat([load_and_clean(p23), load_and_clean(p24)])
    train_df = add_v10_features(train_df)
    test_df = add_v10_features(load_and_clean(p25))
    
    # 2. Train Neural Network
    feats = ['ema_13_slope', 'ema_34_slope', 'ema_55_slope', 'stoch_k', 'stoch_d', 'vol_force', 'atr']
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feats])
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    model.fit(X_train, train_df['target'])
    
    # 3. Predict & Backtest
    test_df['nn_prob'] = model.predict_proba(scaler.transform(test_df[feats]))[:, 1]
    test_df['signal'] = (test_df['nn_prob'].shift(1) > 0.62).astype(int)
    
    equity_curve, trade_log = run_v10_backtest(test_df)
    
    # 4. Output
    print_descriptive_report(equity_curve, trade_log)
    equity_curve.plot(color='seagreen', title="V10 Equity Growth (Unseen 2025 Data)")
    plt.axhline(5000, color='red', linestyle='--', alpha=0.5)
    plt.show()