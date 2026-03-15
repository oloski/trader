import os
import requests
import pandas as pd
import time
from datetime import datetime

# 1. KONFIGURACJA
API_KEY = os.getenv('ALPHA_VANTAGE_KEY')
BASE_URL = 'https://www.alphavantage.co/query'
SAVE_PATH = '/app/shared_data/raw_market_data/'
MAX_RETRIES = 3 # Zabezpieczenie przed pętlą limitów

os.makedirs(SAVE_PATH, exist_ok=True)

def fetch_data(params):
    """Pobiera dane z pętlą while (Fix 3: brak rekurencji)."""
    params['apikey'] = API_KEY
    retries = 0
    
    while retries < MAX_RETRIES:
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "Note" in data:
                print(f"⚠️ Limit API (próba {retries+1}/{MAX_RETRIES}). Czekam 60s...")
                time.sleep(60)
                retries += 1
                continue
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Błąd sieciowy: {e}")
            break
    return None

def calculate_market_indicators(df):
    """Wskaźniki dla OHLC (Giełda/Crypto)."""
    df['dc_high'] = df['high'].rolling(window=20).max()
    df['dc_low'] = df['low'].rolling(window=20).min()
    df['dc_mid'] = (df['dc_high'] + df['dc_low']) / 2
    
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=14).mean()
    
    return df.fillna(0)

def calculate_macro_indicators(df):
    """Wskaźniki dla danych MACRO (Momentum/Trend)."""
    df['momentum'] = df['value'].pct_change()
    df['trend_ma'] = df['value'].rolling(window=7).mean()
    return df.dropna() # Fix 1: usuwamy brakujące, nie wstawiamy 0

def get_crypto_data(symbol):
    print(f"📥 Pobieranie Crypto: {symbol}")
    params = {"function": "DIGITAL_CURRENCY_DAILY", "symbol": symbol, "market": "USD"}
    data = fetch_data(params)
    
    if data and "Time Series (Digital Currency Daily)" in data:
        raw = data["Time Series (Digital Currency Daily)"]
        df = pd.DataFrame.from_dict(raw, orient='index')
        
        # Fix 2: Selekcja po nazwach (bezpieczniejsza niż slicing)
        col_map = {
            '1b. open (USD)': 'open',
            '2b. high (USD)': 'high',
            '3b. low (USD)': 'low',
            '4b. close (USD)': 'close',
            '5. volume': 'volume'
        }
        df = df.rename(columns=col_map)[list(col_map.values())].astype(float)
        return calculate_market_indicators(df.sort_index())
    return None

def get_macro_data(function, name):
    print(f"📥 Pobieranie Macro: {name}")
    data = fetch_data({"function": function})
    
    if data and "data" in data:
        df = pd.DataFrame(data["data"])
        # Fix 1: '.' na NaN i usuwanie braków
        df['value'] = pd.to_numeric(df['value'].replace('.', float('nan')), errors='coerce')
        df = df.dropna().set_index('date').sort_index()
        return calculate_macro_indicators(df)
    return None

def main():
    if not API_KEY:
        print("❌ BŁĄD: Brak ALPHA_VANTAGE_KEY!")
        return

    assets = {
        "crypto": ["BTC", "ETH"],
        "macro": [("FEDERAL_FUNDS_RATE", "FED_RATE"), ("CPI", "INFLATION")]
    }

    for symbol in assets["crypto"]:
        df = get_crypto_data(symbol)
        if df is not None: df.to_csv(f"{SAVE_PATH}{symbol}_daily.csv")
            
    for func, name in assets["macro"]:
        df = get_macro_data(func, name)
        if df is not None: df.to_csv(f"{SAVE_PATH}{name}.csv")

    print(f"✅ Dane gotowe w {SAVE_PATH}")

if __name__ == "__main__":
    main()