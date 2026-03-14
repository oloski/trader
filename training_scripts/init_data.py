import os
import pandas as pd
import pandas_ta as ta
import requests

API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
SYMBOLS = ["NVDA", "BTC", "AAPL", "TSLA"]


def download_and_process():
    path = "/app/shared_data/library"
    os.makedirs(path, exist_ok=True)

    for symbol in SYMBOLS:
        print(f"Pobieranie {symbol}...")
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY&symbol={symbol}"
            f"&outputsize=full&apikey={API_KEY}"
        )
        data = requests.get(url).json()

        ts = data.get("Time Series (Daily)")
        if not ts:
            print(f"  ⚠️  Brak danych dla {symbol}: {data.get('Note') or data.get('Information') or data}")
            continue

        df = pd.DataFrame.from_dict(ts, orient="index").astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)  # od najstarszych

        # Kolumny Alpha Vantage: 1.open 2.high 3.low 4.close 5.volume
        high = df.iloc[:, 1]
        low = df.iloc[:, 2]
        close = df.iloc[:, 3]

        df["ATR"] = ta.atr(high, low, close, length=20)

        donchian = ta.donchian(high, low, length=20)
        df["DC_H"] = donchian.iloc[:, 0]   # DCU
        df["DC_L"] = donchian.iloc[:, 2]   # DCL

        out = f"{path}/{symbol}_enriched.csv"
        df.to_csv(out)
        print(f"  ✅ {symbol} → {out}")

    print("✅ Dane zapisane w shared_data/library")


if __name__ == "__main__":
    download_and_process()
