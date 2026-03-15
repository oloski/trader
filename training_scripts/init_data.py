import os
import pandas as pd
import requests
import time
import sys

API_KEY = os.getenv("ALPHA_VANTAGE_KEY")

ASSETS = {
    "STOCKS": [
        "SPY", "QQQ", "DIA", "IWM", "EWG", "EPOL", "VGK", "EWJ", "FXI", "INDA",
        "NVDA", "AAPL", "TSLA", "AMD", "GLD", "SLV", "PALL", "TLT", "UUP", "VXX"
    ],
    "FOREX": [
        ("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("USD", "PLN"), ("AUD", "USD")
    ],
    "MACRO": [
        "FEDERAL_FUNDS_RATE", "TREASURY_YIELD", "CPI", "UNEMPLOYMENT"
    ],
    "COMMODITIES": [
        "WTI", "BRENT", "NATURAL_GAS", "WHEAT", "CORN", "COFFEE", "SUGAR"
    ], 
    "CRYPTO": ["BTC", "ETH", "SOL"]
}

def calculate_indicators(df, h_col, l_col, c_col):
    df = df.copy()
    for col in [h_col, l_col, c_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[h_col, l_col, c_col]).copy()
    high, low, close = df[h_col], df[l_col], df[c_col]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df.loc[:, 'ATR'] = tr.rolling(window=20).mean()
    df.loc[:, 'DC_H'] = high.rolling(window=20).max()
    df.loc[:, 'DC_L'] = low.rolling(window=20).min()
    return df.dropna()

def fetch_data():
    if not API_KEY:
        print("❌ BŁĄD: Brak klucza API!")
        sys.exit(1)
        
    path = "/app/shared_data/library"
    os.makedirs(path, exist_ok=True)

    for cat, items in ASSETS.items():
        for item in items:
            name = f"{item[0]}{item[1]}" if isinstance(item, tuple) else item
            print(f"📥 Pobieranie {cat}: {name}...")
            
            try:
                if cat == "STOCKS":
                    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={item}&outputsize=full&apikey={API_KEY}"
                elif cat == "FOREX":
                    url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={item[0]}&to_symbol={item[1]}&outputsize=full&apikey={API_KEY}"
                elif cat == "MACRO":
                    # Poprawka dla Treasury Yield
                    maturity = "&maturity=10year" if item == "TREASURY_YIELD" else ""
                    url = f"https://www.alphavantage.co/query?function={item}{maturity}&apikey={API_KEY}"
                elif cat == "COMMODITIES":
                    url = f"https://www.alphavantage.co/query?function={item}&apikey={API_KEY}"
                elif cat == "CRYPTO":
                    url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={item}&market=USD&apikey={API_KEY}"

                r = requests.get(url).json()
                key = next((k for k in r.keys() if any(x in k for x in ["Time Series", "data", "Digital Currency"])), None)

                if key:
                    raw_data = r[key]
                    df = pd.DataFrame(raw_data) if isinstance(raw_data, list) else pd.DataFrame.from_dict(raw_data, orient='index')
                    if 'date' in df.columns: df = df.set_index('date')
                    df = df.sort_index()

                    if cat == "COMMODITIES" or cat == "MACRO":
                        df = calculate_indicators(df, 'value', 'value', 'value')
                    elif cat == "CRYPTO":
                        # Super-bezpieczny selektor kolumn dla Crypto
                        cols = df.columns.tolist()
                        h = [c for c in cols if 'high' in c.lower() and 'usd' in c.lower()]
                        l = [c for c in cols if 'low' in c.lower() and 'usd' in c.lower()]
                        c = [c for c in cols if 'close' in c.lower() and 'usd' in c.lower()]
                        # Jeśli nie znalazł z (USD), weź pierwsze z brzegu high/low/close
                        h_col = h[0] if h else [c for c in cols if 'high' in c.lower()][0]
                        l_col = l[0] if l else [c for c in cols if 'low' in c.lower()][0]
                        c_col = c[0] if c else [c for c in cols if 'close' in c.lower()][0]
                        df = calculate_indicators(df, h_col, l_col, c_col)
                    elif cat == "STOCKS":
                        df = calculate_indicators(df, df.columns[1], df.columns[2], df.columns[4])
                    else: # FOREX
                        df = calculate_indicators(df, df.columns[1], df.columns[2], df.columns[3])
                    
                    df.to_csv(f"{path}/{name}_enriched.csv")
                    print(f"   ✅ Zapisano {name} ({len(df)} rekordów)")
                else:
                    print(f"   ❌ Pominięto {item}: {r.get('Note', r.get('Error Message', 'Brak danych'))}")
            except Exception as e:
                print(f"   ❌ Błąd krytyczny dla {item}: {e}")
            time.sleep(1.0)

if __name__ == "__main__":
    fetch_data()
    print("\n🎯 BIBLIOTEKA DANYCH KOMPLETNA.")