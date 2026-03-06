import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

# ===============================
# INIT MT5
# ===============================
def init_mt5():
    account = os.getenv("MT5_ACCOUNT")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
    
    if not account or not password:
        raise RuntimeError("Missing MT5 credentials in .env file.")

    if not mt5.initialize(login=int(account), password=password, server=server):
        raise RuntimeError(f"MT5 init failed -> {mt5.last_error()}")
    
    print("✅ MT5 initialized successfully")

# ===============================
# FETCH DATA
# ===============================
def fetch_data(symbol, year, timeframe=mt5.TIMEFRAME_H4):
    TIMEZONE = pytz.UTC
    from_date = datetime(year, 1, 1, tzinfo=TIMEZONE)
    to_date   = datetime(year, 12, 31, tzinfo=TIMEZONE)
    
    print(f"📡 Fetching {symbol} data for year {year}...")
    rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)
    
    if rates is None or len(rates) == 0:
        print(f"⚠️ No data returned from MT5 for {symbol} in {year}")
        return None
        
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    
    df = df[["time", "open", "high", "low", "close", "volume", "spread"]]
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Historical MT5 Data.")
    parser.add_argument("--symbol", type=str, default="WTIUSD", help="Asset symbol (e.g., WTIUSD, XAUUSD)")
    parser.add_argument("--year", type=int, default=2024, help="Year to download data for")
    parser.add_argument("--output", type=str, help="Output CSV filename. E.g., data/raw/WTIUSD_2024.csv")
    args = parser.parse_args()
    
    try:
        init_mt5()
        df = fetch_data(args.symbol.upper(), args.year)
        if df is not None:
            output_file = args.output or f"data/raw/{args.symbol.upper()}_{args.year}.csv"
            df.to_csv(output_file, index=False)
            print(f"💾 Saved {len(df)} rows to {output_file}")
    finally:
        mt5.shutdown()
