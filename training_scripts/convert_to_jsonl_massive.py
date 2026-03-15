import pandas as pd
import json
import os

# --- KONFIGURACJA ---
INPUT_DIR = "/app/shared_data/raw_market_data/"
OUTPUT_FILE = "/app/shared_data/market_training.jsonl"


def build_market_output(asset_name: str, row: pd.Series, change: float) -> str:
    """Buduje ustrukturyzowany output z sygnałem i kontekstem wskaźników."""
    atr    = row.get('atr', 0)
    dc_h   = row.get('dc_high', 0)
    dc_l   = row.get('dc_low', 0)
    close  = row.get('close', 0)
    rsi    = row.get('rsi', 0)
    adx    = row.get('adx', 0)
    macd_h = row.get('macd_hist', 0)

    # Sygnał Donchiana (strategia Żółwia)
    if dc_h and dc_l:
        if close >= dc_h:
            signal = f"BUY — wybicie szczytu kanału Donchiana (DC_H={dc_h:.2f})."
        elif close <= dc_l:
            signal = f"SELL — przebicie dołka kanału Donchiana (DC_L={dc_l:.2f})."
        else:
            signal = f"HOLD — cena wewnątrz kanału [{dc_l:.2f}–{dc_h:.2f}]."
    else:
        signal = "HOLD — brak danych kanału."

    # Kontekst wskaźników
    trend = "wzrostowy" if macd_h > 0 else "spadkowy"
    strength = "silny" if adx > 25 else "słaby"
    rsi_ctx = "wykupienie" if rsi > 70 else ("wyprzedanie" if rsi < 30 else "neutralny")

    output = (
        f"{asset_name}: {signal} "
        f"Zmiana: {change:+.2f}%, ATR={atr:.4f}. "
        f"Trend {trend} ({strength}, ADX={adx:.1f}). "
        f"RSI={rsi:.1f} ({rsi_ctx}), MACD_hist={macd_h:.4f}."
    )
    return output


def build_macro_output(asset_name: str, row: pd.Series, change: float) -> str:
    """Buduje output dla danych makroekonomicznych."""
    value    = row.get('value', 0)
    ma7      = row.get('trend_ma7', 0)   # fix: było trend_ma, teraz trend_ma7
    ma30     = row.get('trend_ma30', 0)
    zscore   = row.get('zscore_1y', 0)

    output = f"Wskaźnik {asset_name} wynosi {value:.2f}"
    if ma7:
        output += f" (MA7={ma7:.2f}, MA30={ma30:.2f})"
    output += ". "

    if zscore > 1.5:
        output += f"Historycznie wysoki poziom (z-score={zscore:.2f}) — potencjalna presja na aktywa ryzykowne."
    elif zscore < -1.5:
        output += f"Historycznie niski poziom (z-score={zscore:.2f}) — sprzyja aktywom ryzykownym."
    elif change > 0.5:
        output += "Wzrost dynamiki — obserwuj wpływ na rynki."
    elif change < -0.5:
        output += "Spadek dynamiki — potencjalne rozluźnienie presji rynkowej."
    else:
        output += "Stabilizacja wskaźnika — brak wyraźnego impulsu."
    return output


def process_csv_to_jsonl():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Folder {INPUT_DIR} nie istnieje! Uruchom najpierw init_data.py.")
        return

    all_entries = []

    for filename in sorted(os.listdir(INPUT_DIR)):
        if not filename.endswith(".csv"):
            continue

        file_path = os.path.join(INPUT_DIR, filename)
        asset_name = filename.replace("_daily.csv", "").replace(".csv", "")

        # Fix 1: index_col=0 — daty trafiają do indeksu, nie do Unnamed:0
        try:
            df = pd.read_csv(file_path, index_col=0)
        except Exception as e:
            print(f"  ❌ Błąd odczytu {filename}: {e}")
            continue

        print(f"📊 Przetwarzanie: {asset_name} ({len(df)} wierszy)...")

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            date = df.index[i]

            try:
                if 'close' in df.columns:
                    # Fix 3: zabezpieczenie przed dzieleniem przez zero
                    prev_close = prev_row['close']
                    if not prev_close or prev_close == 0:
                        continue
                    change = ((row['close'] - prev_close) / prev_close) * 100
                    context = (
                        f"Zamknięcie: {row['close']:.2f} USD, "
                        f"Zmiana: {change:+.2f}%, "
                        f"Wolumen: {row.get('volume', 'N/A')}, "
                        f"ATR: {row.get('atr', 0):.4f}, "
                        f"DC: [{row.get('dc_low', 0):.2f}–{row.get('dc_high', 0):.2f}]"
                    )
                    instruction = f"Przeanalizuj sytuację techniczną na {asset_name} na dzień {date}."
                    output = build_market_output(asset_name, row, change)

                elif 'value' in df.columns:
                    change = row.get('momentum', 0) * 100
                    context = (
                        f"Wartość: {row['value']:.2f}, "
                        f"Dynamika: {change:+.2f}%, "
                        f"MA7: {row.get('trend_ma', 0):.2f}"
                    )
                    instruction = f"Jakie znaczenie dla rynku mają dane makroekonomiczne {asset_name} z dnia {date}?"
                    output = build_macro_output(asset_name, row, change)

                else:
                    # Fix 2: nieznany format — pomijamy zamiast crashować
                    continue

            except Exception as e:
                print(f"  ⚠️  Wiersz {i} w {filename}: {e}")
                continue

            all_entries.append({
                "instruction": instruction,
                "input": f"Data: {date} | {context}",
                "output": output
            })

    if not all_entries:
        print("⚠️  Brak danych do zapisu.")
        return

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n✅ Market Dataset gotowy: {len(all_entries)} rekordów → {OUTPUT_FILE}")


if __name__ == "__main__":
    process_csv_to_jsonl()
