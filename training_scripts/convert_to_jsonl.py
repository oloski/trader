"""
Konwertuje dane CSV (Alpha Vantage + wskaźniki) oraz PDF z biblioteki
do formatu JSONL z instrukcjami (instruction-tuning) pod strategię Żółwia.

Wynikowy plik: /app/shared_data/library/training_data.jsonl
"""

import os
import glob
import json
import textwrap

import pandas as pd

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  PyMuPDF niedostępny — pomijam ekstrakcję PDF.")

LIBRARY = "/app/shared_data/library"
OUT_FILE = os.path.join(LIBRARY, "training_data.jsonl")

# ── helpers ──────────────────────────────────────────────────────────────────

def csv_to_instructions(csv_path: str) -> list[dict]:
    """Zamienia wiersze CSV na przykłady instrukcji w stylu Alpaca."""
    symbol = os.path.basename(csv_path).replace("_enriched.csv", "")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).dropna()

    records = []
    for i in range(20, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Kontekst rynkowy
        instruction = (
            f"Jesteś traderem stosującym strategię Żółwia. "
            f"Analizujesz {symbol} na dzień {df.index[i].date()}. "
            f"Close={row.iloc[3]:.2f}, ATR={row['ATR']:.4f}, "
            f"DC_H={row['DC_H']:.2f}, DC_L={row['DC_L']:.2f}. "
            f"Poprzedni close={prev.iloc[3]:.2f}. "
            f"Podaj sygnał transakcyjny (BUY / SELL / HOLD) z krótkim uzasadnieniem."
        )

        # Prosta reguła żółwia jako odpowiedź wzorcowa
        close = row.iloc[3]
        if close >= row["DC_H"]:
            signal = "BUY"
            reason = "Cena wybija szczyt kanału Donchiana (DC_H) — sygnał wejścia long."
        elif close <= row["DC_L"]:
            signal = "SELL"
            reason = "Cena przebija dołek kanału Donchiana (DC_L) — sygnał wejścia short / zamknięcia long."
        else:
            signal = "HOLD"
            reason = "Cena wewnątrz kanału Donchiana — brak sygnału kierunkowego."

        output = f"{signal}\n{reason}\nWielkość pozycji: 1% kapitału / ATR ({row['ATR']:.4f})."

        records.append({"instruction": instruction, "input": "", "output": output})

    return records


def pdf_to_chunks(pdf_path: str, chunk_chars: int = 800) -> list[dict]:
    """Dzieli tekst PDF na fragmenty i tworzy przykłady Q&A."""
    if not PDF_SUPPORT:
        return []

    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    # Prosty podział na fragmenty
    chunks = textwrap.wrap(full_text, chunk_chars, break_long_words=False)
    title = os.path.basename(pdf_path)

    records = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 100:
            continue
        records.append({
            "instruction": (
                f"Poniżej fragment książki inwestycyjnej '{title}'. "
                "Streść główną myśl lub zasadę handlową zawartą w tym fragmencie."
            ),
            "input": chunk,
            "output": "",   # pole pozostaje puste — model generuje samodzielnie
        })

    return records


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    all_records: list[dict] = []

    # 1. Dane CSV (sygnały żółwia)
    csv_files = glob.glob(os.path.join(LIBRARY, "*_enriched.csv"))
    if not csv_files:
        print("⚠️  Brak plików CSV — najpierw uruchom init_data.py.")
    for csv_path in sorted(csv_files):
        print(f"  CSV → {os.path.basename(csv_path)}")
        all_records.extend(csv_to_instructions(csv_path))

    # 2. Książki PDF z katalogu books/
    books_dir = os.path.join(LIBRARY, "books")
    pdf_files = glob.glob(os.path.join(books_dir, "*.pdf"))
    for pdf_path in sorted(pdf_files):
        print(f"  PDF → {os.path.basename(pdf_path)}")
        all_records.extend(pdf_to_chunks(pdf_path))

    # 3. Zapis JSONL
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Zapisano {len(all_records)} przykładów → {OUT_FILE}")


if __name__ == "__main__":
    main()
