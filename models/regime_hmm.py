import pandas as pd
import numpy as np
from hmmlearn import hmm
import warnings

warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
FILE_NAME = "xau_1d_latest.csv"

def run_regime_audit():
    try:
        df = pd.read_csv(FILE_NAME, sep='\t')
        df.columns = [c.replace('<', '').replace('>', '').lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # FEATURES: Log Returns (Direction) & ATR % (Volatility)
        df['h_l'] = df['high'] - df['low']
        df['atr'] = df['h_l'].rolling(14).mean()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['atr_pct'] = df['atr'] / df['close'] 
        
        df = df.dropna()

        # TRAIN HMM (3 States)
        X = df[['log_ret', 'atr_pct']].values
        model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X)
        df['regime'] = model.predict(X)

        # LOGIC: Mapping States based on ATR and Returns Variance
        # sorted_vol: [Low, Mid, High]
        vol_means = model.means_[:, 1]
        sorted_vol = np.argsort(vol_means)
        
        # We define "Sideways" as Mid-Vol with low absolute returns
        # and "High Vol" as the state with the highest ATR mean.
        label_map = {
            sorted_vol[0]: "📉 LOW VOL (Slow Accumulation/Quiet)",
            sorted_vol[1]: "↔️ SIDEWAYS (Rangebound/Mean Reversion)",
            sorted_vol[2]: "🌪️ HIGH VOL (Expansion/Major Breakouts)"
        }

        latest = df.iloc[-1]
        curr_id = int(latest['regime'])
        
        print("\n" + "✅" * 15)
        print(f"📊 XAUUSD REGIME: {label_map[curr_id]}")
        print(f"Price: {latest['close']:.2f} | ATR %: {latest['atr_pct']:.4%}")
        print("✅" * 15)

        # Nonchalant Audit Tip
        if curr_id == sorted_vol[2]:
            print("\nBro, the ATR is peaking. Expect high slippage on M5.")
        elif curr_id == sorted_vol[1]:
            print("\nSideways vibes. Your 1.2x ATR breakout logic might get fake-outs here.")

    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    run_regime_audit()