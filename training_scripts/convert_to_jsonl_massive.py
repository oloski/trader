import pandas as pd
import os
import json

LIBRARY_PATH = "/app/shared_data/library"
OUTPUT_FILE = "/app/shared_data/market_training.jsonl"

def find_close_column(df):
    """Próbuje znaleźć kolumnę z ceną zamknięcia niezależnie od nazwy"""
    possible_names = ['close', 'value', '4. close', '4a. close (usd)', 'adjusted close']
    cols = {c.lower(): c for c in df.columns}
    for name in possible_names:
        if name in cols:
            return cols[name]
    # Jeśli nie znalazł, weź pierwszą dostępną kolumnę, która nie jest indeksem/wskaźnikiem
    skip = ['index', 'date', 'atr', 'dc_h', 'dc_l']
    for c in df.columns:
        if c.lower() not in skip:
            return c
    return None

def generate_market_prompts():
    all_entries = []
    if not os.path.exists(LIBRARY_PATH):
        print(f"❌ Folder {LIBRARY_PATH} nie istnieje!")
        return

    files = [f for f in os.listdir(LIBRARY_PATH) if f.endswith("_enriched.csv")]
    
    for file in files:
        asset_name = file.replace("_enriched.csv", "")
        df = pd.read_csv(os.path.join(LIBRARY_PATH, file))
        
        close_col = find_close_column(df)
        if not close_col:
            print(f"⚠️ Pominąłem {file}: nie znaleziono kolumny ceny.")
            continue

        print(f"Processing {asset_name} (Column: {close_col})...")

        # Generowanie próbek co 5 dni (stride=5), by uniknąć overfittingu na Blackwellu
        for i in range(20, len(df), 5):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            close_val = float(row[close_col])
            prev_close = float(prev_row[close_col])
            atr = float(row['ATR'])
            dc_h = float(row['DC_H'])
            dc_l = float(row['DC_L'])

            instruction = f"Analiza trendu i zmienności dla {asset_name}."
            input_data = (f"Cena: {close_val:.2f}, Poprzednia: {prev_close:.2f}, "
                          f"ATR: {atr:.2f}, Donchian H: {dc_h:.2f}, L: {dc_l:.2f}")
            
            # Prosta logika opisu dla modelu
            trend = "wzrostowy (powyżej DC_H)" if close_val >= dc_h else \
                    "spadkowy (poniżej DC_L)" if close_val <= dc_l else "boczny"
            
            output = f"Na podstawie danych, {asset_name} wykazuje trend {trend}. Poziom zmienności ATR wynosi {atr:.2f}."
            
            all_entries.append({
                "instruction": instruction,
                "input": input_data,
                "output": output
            })

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Sukces! Wygenerowano {len(all_entries)} próbek rynkowych w {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_market_prompts()