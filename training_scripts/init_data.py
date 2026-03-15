import os
import sys
import requests
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- KONFIGURACJA ---
API_KEY = os.getenv('ALPHA_VANTAGE_KEY')
BASE_URL = 'https://www.alphavantage.co/query'
SAVE_PATH = '/app/shared_data/raw_market_data/'
MAX_RETRIES = 3

ASSETS = {
    "STOCKS": [
        # Główne indeksy USA
        "SPY", "QQQ", "DIA", "IWM",
        # Wszystkie 11 sektorów SPDR — rotacja sektorowa
        "XLK", "XLF", "XLE", "XLV", "XLB", "XLI", "XLU", "XLP", "XLY", "XLRE", "XLC",
        # Obligacje (krótki/średni/długi koniec krzywej)
        "SHY", "IEF", "TLT",
        # Rynki globalne i surowce
        "EEM", "EWZ", "FXI",        # EM: szeroki, Brazylia, Chiny
        "GLD", "SLV", "PDBC",       # Złoto, Srebro, zdywersyfikowane surowce
        "UUP",                       # US Dollar Index ETF — proxy DXY
        "VNQ",                       # Real Estate REIT
        # Zmienność
        "VXX",                       # VIX futures ETN
        # Giganci Tech + Finanse
        "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "JPM", "BRK-B",
    ],
    "FOREX": [
        # G7
        ("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"),
        ("AUD", "USD"), ("USD", "CAD"), ("USD", "CHF"), ("NZD", "USD"),
        # Krzyżowe
        ("EUR", "JPY"), ("GBP", "JPY"), ("EUR", "GBP"),
        # Rynki wschodzące
        ("USD", "CNY"),  # Yuan — kluczowy dla surowców i EM
        ("USD", "PLN"),  # Złoty
        ("USD", "MXN"),  # Peso — barometr risk-on/off EM
    ],
    "COMMODITIES": [
        "WTI", "BRENT", "NATURAL_GAS",    # Energetyka
        "COPPER", "ALUMINUM",              # Metale przemysłowe (GOLD/SILVER → ETF w STOCKS: GLD/SLV)
        "WHEAT", "CORN", "COTTON", "SUGAR", "COFFEE",  # Rolnictwo
    ],
    "CRYPTO": [
        # Alpha Vantage obsługuje tylko wybrane krypto
        "BTC", "ETH", "SOL", "XRP",       # BNB/AVAX/LINK niedostępne w tym API
    ],
    "MACRO": [
        ("REAL_GDP",           "REAL_GDP"),          # PKB — najważniejszy wskaźnik
        ("FEDERAL_FUNDS_RATE", "FED_RATE"),          # Stopa Fed
        ("CPI",                "INFLATION"),         # Inflacja (CPI)
        ("INFLATION",          "INFLATION_YOY"),     # Inflacja r/r (roczna)
        ("NONFARM_PAYROLL",    "NFP"),               # Payrolls — najsilniejszy rynkowy ruch
        ("UNEMPLOYMENT",       "UNEMPLOYMENT"),
        ("TREASURY_YIELD",     "TREASURY_10Y"),      # Rentowność 10Y
        ("RETAIL_SALES",       "RETAIL_SALES"),
        ("DURABLES",           "DURABLES"),          # Zamówienia dóbr trwałych — leading indicator
    ],
}

# Łączna liczba requestów: ~30 STOCKS + 13 FOREX + 11 COMMODITIES + 7 CRYPTO + 9 MACRO = ~70
# Premium 75/min + 8 wątków → cały dataset w ~2-3 minuty

os.makedirs(SAVE_PATH, exist_ok=True)




def calculate_market_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wskaźniki techniczne dla danych OHLC (Stocks/Forex/Crypto).

    Trend:        EMA20, EMA50, EMA200, MACD (12/26/9)
    Zmienność:    ATR14, Bollinger Bands (20, 2σ)
    Momentum:     RSI14, ROC10
    Siła trendu:  ADX14
    Kanały:       Donchian (20)
    Wolumen:      Volume MA20, OBV (jeśli dostępny wolumen)
    """
    close = df['close']
    high  = df['high']
    low   = df['low']

    # --- Donchian Channels (strategia Żółwia) ---
    df['dc_high'] = high.rolling(20).max()
    df['dc_low']  = low.rolling(20).min()
    df['dc_mid']  = (df['dc_high'] + df['dc_low']) / 2

    # --- ATR ---
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # --- EMA (filtr trendu) ---
    df['ema20']  = close.ewm(span=20,  adjust=False).mean()
    df['ema50']  = close.ewm(span=50,  adjust=False).mean()
    df['ema200'] = close.ewm(span=200, adjust=False).mean()

    # --- MACD (12/26/9) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    # --- RSI 14 ---
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float('nan'))
    df['rsi'] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands (20, 2σ) ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid  # znormalizowana szerokość

    # --- ROC 10 (Rate of Change) ---
    df['roc10'] = close.pct_change(10) * 100

    # --- ADX 14 (siła trendu — kluczowy dla Żółwia) ---
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr14     = tr.rolling(14).mean()
    plus_di   = 100 * (plus_dm.rolling(14).mean()  / atr14.replace(0, float('nan')))
    minus_di  = 100 * (minus_dm.rolling(14).mean() / atr14.replace(0, float('nan')))
    dx        = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float('nan')))
    df['adx']       = dx.rolling(14).mean()
    df['plus_di']   = plus_di
    df['minus_di']  = minus_di

    # --- Volume MA20 i OBV (tylko jeśli kolumna volume istnieje) ---
    if 'volume' in df.columns:
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        obv = (df['volume'] * close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).cumsum()
        df['obv'] = obv

    return df.fillna(0)


def calculate_macro_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wskaźniki dla danych jednowymiarowych (Macro/Commodities).

    Trend:    MA7, MA30
    Momentum: pct_change (1d), pct_change (30d)
    Odchylenie: z-score od 252-dniowej średniej (roczny kontekst)
    """
    v = df['value']
    df['momentum']    = v.pct_change()
    df['momentum_30'] = v.pct_change(30)
    df['trend_ma7']   = v.rolling(7).mean()
    df['trend_ma30']  = v.rolling(30).mean()
    # Z-score: ile odchyleń standardowych od rocznej średniej
    ma252  = v.rolling(252).mean()
    std252 = v.rolling(252).std()
    df['zscore_1y'] = (v - ma252) / std252.replace(0, float('nan'))
    return df.dropna()


def _parse_ohlc(raw: dict, col_map: dict, symbol: str) -> pd.DataFrame | None:
    """Wspólna logika parsowania odpowiedzi OHLC."""
    df = pd.DataFrame.from_dict(raw, orient='index')
    missing = set(col_map.keys()) - set(df.columns)
    if missing:
        print(f"  ⚠️  Nieoczekiwane kolumny API dla {symbol}: {set(df.columns)}")
        return None
    df = df.rename(columns=col_map)[list(col_map.values())].astype(float)
    return df.sort_index()


def get_stock_data(symbol: str) -> pd.DataFrame | None:
    print(f"  📥 STOCK: {symbol}")
    data = fetch_data({"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "full"})
    if not data or "Time Series (Daily)" not in data:
        print(f"    ❌ Brak danych dla {symbol}")
        return None
    col_map = {
        "1. open": "open", "2. high": "high",
        "3. low": "low",   "4. close": "close", "5. volume": "volume",
    }
    df = _parse_ohlc(data["Time Series (Daily)"], col_map, symbol)
    return calculate_market_indicators(df) if df is not None else None


def get_forex_data(from_sym: str, to_sym: str) -> pd.DataFrame | None:
    name = f"{from_sym}{to_sym}"
    print(f"  📥 FOREX: {name}")
    data = fetch_data({
        "function": "FX_DAILY", "from_symbol": from_sym,
        "to_symbol": to_sym, "outputsize": "full",
    })
    ts_key = "Time Series FX (Daily)"
    if not data or ts_key not in data:
        print(f"    ❌ Brak danych dla {name}")
        return None
    col_map = {
        "1. open": "open", "2. high": "high",
        "3. low": "low",   "4. close": "close",
    }
    df = _parse_ohlc(data[ts_key], col_map, name)
    return calculate_market_indicators(df) if df is not None else None


def get_commodity_data(function: str) -> pd.DataFrame | None:
    print(f"  📥 COMMODITY: {function}")
    data = fetch_data({"function": function, "interval": "daily"})
    if not data or "data" not in data:
        print(f"    ❌ Brak danych dla {function}")
        return None
    df = pd.DataFrame(data["data"])
    df['value'] = pd.to_numeric(df['value'].replace('.', float('nan')), errors='coerce')
    df = df.dropna().set_index('date').sort_index()
    return calculate_macro_indicators(df)


def get_crypto_data(symbol: str) -> pd.DataFrame | None:
    print(f"  📥 CRYPTO: {symbol}")
    data = fetch_data({"function": "DIGITAL_CURRENCY_DAILY", "symbol": symbol, "market": "USD"})
    ts_key = "Time Series (Digital Currency Daily)"
    if not data or ts_key not in data:
        print(f"    ❌ Brak danych dla {symbol}")
        return None
    col_map = {
        "1. open": "open", "2. high": "high",
        "3. low": "low",   "4. close": "close", "5. volume": "volume",
    }
    df = _parse_ohlc(data[ts_key], col_map, symbol)
    return calculate_market_indicators(df) if df is not None else None


# --- Rate limiter dla premium: 75 req/min ---
RATE_LIMIT     = 75          # requestów na minutę
MIN_INTERVAL   = 60 / RATE_LIMIT  # 0.8s między requestami
MAX_WORKERS    = 8           # wątki równoległe (bezpieczne przy 75/min)
_rate_lock     = threading.Lock()
_last_req_time = 0.0


def _rate_wait() -> None:
    """Globalny token bucket — gwarantuje max 75 req/min."""
    global _last_req_time
    with _rate_lock:
        elapsed = time.monotonic() - _last_req_time
        wait    = MIN_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)
        _last_req_time = time.monotonic()


def fetch_data(params: dict) -> dict | None:
    """Pobiera dane z Alpha Vantage z rate limiterem i retry."""
    params = dict(params)
    params['apikey'] = API_KEY
    for attempt in range(1, MAX_RETRIES + 1):
        _rate_wait()
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if "Note" in data:
                print(f"  ⚠️  Limit API (próba {attempt}/{MAX_RETRIES}). Czekam 15s...")
                time.sleep(15)
                continue
            if "Error Message" in data:
                print(f"  ❌ Błąd API: {data['Error Message']}")
                return None
            return data
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Błąd sieciowy: {e}")
            break
    return None


def save(df: pd.DataFrame | None, name: str) -> None:
    if df is not None:
        df.to_csv(f"{SAVE_PATH}{name}.csv")
        print(f"    ✅ {name} ({len(df)} rekordów)")


def _fetch_and_save(task: tuple) -> str:
    """Zadanie dla wątku: (typ, args...). Zwraca nazwę aktywa."""
    kind = task[0]
    if kind == "stock":
        sym = task[1]
        save(get_stock_data(sym), sym)
        return sym
    elif kind == "forex":
        f, t = task[1], task[2]
        save(get_forex_data(f, t), f"{f}{t}")
        return f"{f}{t}"
    elif kind == "commodity":
        func = task[1]
        save(get_commodity_data(func), func)
        return func
    elif kind == "crypto":
        sym = task[1]
        save(get_crypto_data(sym), f"{sym}_daily")
        return sym
    elif kind == "macro":
        func, name = task[1], task[2]
        params = {"function": func}
        if func == "TREASURY_YIELD":
            params["maturity"] = "10year"
        data = fetch_data(params)
        if data and "data" in data:
            df = pd.DataFrame(data["data"])
            df['value'] = pd.to_numeric(df['value'].replace('.', float('nan')), errors='coerce')
            df = df.dropna().set_index('date').sort_index()
            save(calculate_macro_indicators(df), name)
        else:
            print(f"    ❌ Brak danych dla {name}")
        return name
    return "unknown"


def run_parallel(tasks: list[tuple], label: str) -> None:
    print(f"{label} ({len(tasks)} aktywów, {MAX_WORKERS} wątki):")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_and_save, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"  ⚠️  Błąd wątku: {e}")


def run_stocks():
    run_parallel([("stock", sym) for sym in ASSETS["STOCKS"]], "📈 STOCKS")

def run_forex():
    run_parallel([("forex", f, t) for f, t in ASSETS["FOREX"]], "💱 FOREX")

def run_commodities():
    run_parallel([("commodity", func) for func in ASSETS["COMMODITIES"]], "🛢️  COMMODITIES")

def run_crypto():
    run_parallel([("crypto", sym) for sym in ASSETS["CRYPTO"]], "₿  CRYPTO")

def run_macro():
    run_parallel([("macro", func, name) for func, name in ASSETS["MACRO"]], "📉 MACRO")


RUNNERS = {
    "STOCKS":      run_stocks,
    "FOREX":       run_forex,
    "COMMODITIES": run_commodities,
    "CRYPTO":      run_crypto,
    "MACRO":       run_macro,
}


def main():
    if not API_KEY:
        print("❌ BŁĄD: Brak ALPHA_VANTAGE_KEY!")
        return

    # Uruchom tylko wybraną kategorię lub wszystkie
    # Przykład: python init_data.py STOCKS
    #           python init_data.py FOREX MACRO
    categories = [c.upper() for c in sys.argv[1:]] if len(sys.argv) > 1 else list(RUNNERS)
    unknown = [c for c in categories if c not in RUNNERS]
    if unknown:
        print(f"❌ Nieznane kategorie: {unknown}. Dostępne: {list(RUNNERS)}")
        return

    total = sum(len(ASSETS[c]) for c in categories)
    est_sec = (total / MAX_WORKERS) * MIN_INTERVAL
    print(f"🚀 Pobieranie {total} aktywów: {categories}")
    print(f"   Szacowany czas (premium 75/min, {MAX_WORKERS} wątków): ~{est_sec:.0f}s\n")

    for cat in categories:
        RUNNERS[cat]()
        print()

    print(f"✅ Dane gotowe w {SAVE_PATH}")


if __name__ == "__main__":
    main()
