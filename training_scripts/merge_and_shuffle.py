import json
import random
import os

# --- KONFIGURACJA ---
MARKET_DATA_PATH = "/app/shared_data/market_training.jsonl"
BOOKS_DATA_PATH = "/app/shared_data/books_training.jsonl"
OUTPUT_FILE = "/app/shared_data/master_training.jsonl"

MARKET_MULTIPLIER = 3
SEED = 42
REQUIRED_FIELDS = {"instruction", "input", "output"}

random.seed(SEED)


def load_jsonl(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        print(f"⚠️  Plik {file_path} nie istnieje. Pomijam.")
        return []

    data, skipped = [], 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Linia {i}: błąd JSON — {e}")
                skipped += 1
                continue

            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                print(f"  ⚠️  Linia {i}: brak pól {missing} — pomijam rekord.")
                skipped += 1
                continue

            data.append(record)

    if skipped:
        print(f"  ℹ️  Pominięto {skipped} nieprawidłowych rekordów w {os.path.basename(file_path)}")
    return data


def main():
    print("🔄 Ładowanie danych do fuzji...")

    market_data = load_jsonl(MARKET_DATA_PATH)
    books_data = load_jsonl(BOOKS_DATA_PATH)

    if not market_data and not books_data:
        print("❌ Brak danych do połączenia! Sprawdź pliki wejściowe.")
        return

    print(f"\n📊 Statystyki przed fuzją:")
    print(f"   - Dane rynkowe (unikalne): {len(market_data)}")
    print(f"   - Dane z książek:          {len(books_data)}")

    # Upsampling przez deep copy — unikamy shallow copy przy mnożeniu listy
    balanced_market = [dict(entry) for entry in market_data * MARKET_MULTIPLIER]

    combined_dataset = balanced_market + books_data
    total = len(combined_dataset)

    print(f"\n🔀 Mieszanie danych (seed={SEED})...")
    random.shuffle(combined_dataset)

    print(f"💾 Zapisywanie Master Datasetu → {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in combined_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    market_pct = len(balanced_market) / total * 100 if total else 0
    books_pct = len(books_data) / total * 100 if total else 0

    print(f"\n✅ Sukces!")
    print(f"   Łączna liczba rekordów:  {total}")
    print(f"   Dane rynkowe (x{MARKET_MULTIPLIER}): {len(balanced_market):>6}  ({market_pct:.1f}%)")
    print(f"   Dane z książek:          {len(books_data):>6}  ({books_pct:.1f}%)")
    print(f"\n🚀 Możesz teraz uruchomić train_blackwell_v2.py")


if __name__ == "__main__":
    main()
