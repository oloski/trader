import fitz  # PyMuPDF
import os
import json
import re
import random
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# --- KONFIGURACJA ---
BOOKS_PATH = "/app/shared_data/library/books"
OUTPUT_FILE = "/app/shared_data/books_training.jsonl"
CHUNK_SIZE = 2500

# Opcja A: ustaw klucz jeśli chcesz generować output przez Claude API
# Opcja C (domyślna): ekstrakcja lokalna — klucz niepotrzebny
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

random.seed(42)

INSTRUCTIONS = [
    "Przeanalizuj poniższy fragment pod kątem strategii tradingowej i wyciągnij kluczowe wnioski.",
    "Na podstawie tego tekstu wyjaśnij jak profesjonalny trader powinien zareagować w opisanej sytuacji.",
    "Podsumuj zasady zarządzania ryzykiem lub psychologii zawarte w tym fragmencie.",
    "Zidentyfikuj najważniejsze pojęcia techniczne w poniższym tekście i wyjaśnij ich znaczenie.",
    "Jakie lekcje płyną z tego tekstu dla algorytmicznego systemu transakcyjnego?",
]

# Słowa kluczowe — zdania je zawierające trafią do outputu
TRADING_KEYWORDS = [
    "risk", "strategy", "position", "trend", "breakout", "stop", "loss",
    "profit", "entry", "exit", "volume", "signal", "momentum", "support",
    "resistance", "capital", "drawdown", "volatility", "ryzyk", "strategia",
    "pozycja", "trend", "sygnał", "kapital", "straty", "zysk",
]


def clean_text(text: str) -> str:
    """Usuwa numery stron, nadmiarowe białe znaki i uszkodzone znaki Unicode."""
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u200b', '').replace('\uf0b7', '')
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Bezpieczny podział na zdania — nie niszczy skrótów ani liczb."""
    # Rozpoznaje koniec zdania tylko gdy po kropce/!/? jest spacja i wielka litera
    return re.split(r'(?<=[.!?])\s+(?=[A-ZŁŚĆŃÓŻ])', text)


def smart_chunking(text: str, max_chars: int) -> list[str]:
    """Dzieli tekst na fragmenty nie przecinając zdań. Pomija puste chunki."""
    sentences = split_sentences(text)
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) < max_chars:
            current += sentence + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence + " "

    if current.strip():
        chunks.append(current.strip())
    return chunks


def extract_key_sentences(chunk: str, max_sentences: int = 4) -> list[str]:
    """Opcja C: wyciąga zdania zawierające słowa kluczowe tradingowe."""
    sentences = split_sentences(chunk)
    key = [
        s.strip() for s in sentences
        if any(kw in s.lower() for kw in TRADING_KEYWORDS) and len(s.strip()) > 40
    ]
    return key[:max_sentences]


def generate_output_local(instruction: str, chunk: str) -> str:
    """
    Opcja C — ekstrakcja lokalna.
    Buduje ustrukturyzowaną odpowiedź z kluczowych zdań chunka.
    Nie wymaga API ani modelu — działa od razu.
    """
    key_sentences = extract_key_sentences(chunk)

    if not key_sentences:
        # Fallback: weź pierwsze dwa zdania jako kontekst
        key_sentences = split_sentences(chunk)[:2]

    response = f"{instruction}\n\nNa podstawie analizy tekstu można wyciągnąć następujące wnioski:\n\n"
    for i, s in enumerate(key_sentences, 1):
        response += f"{i}. {s}\n"
    response += "\nPrincypia te są kluczowe dla budowy skutecznego systemu tradingowego."
    return response


def generate_output_api(instruction: str, chunk: str) -> str:
    """
    Opcja A — generowanie przez Claude API.
    Aktywuje się gdy ANTHROPIC_API_KEY jest ustawiony w środowisku.
    Wymaga: pip install anthropic
    """
    try:
        import anthropic  # type: ignore  # instalowane wewnątrz Dockera
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",  # najtańszy, wystarczający do syntezy
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"{instruction}\n\n---\n{chunk}\n---"
            }]
        )
        return message.content[0].text
    except Exception as e:
        print(f"  ⚠️  API error, fallback do lokalnego: {e}")
        return generate_output_local(instruction, chunk)


def generate_output(instruction: str, chunk: str) -> str:
    """Wybiera metodę generowania outputu na podstawie dostępności klucza API."""
    if ANTHROPIC_API_KEY:
        return generate_output_api(instruction, chunk)
    return generate_output_local(instruction, chunk)


def extract_text_from_epub(file_path: str) -> str | None:
    try:
        book = epub.read_epub(file_path)
        text_content = ""
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'lxml')
                text_content += soup.get_text() + "\n"
        return text_content
    except Exception as e:
        print(f"  ❌ Błąd EPUB {file_path}: {e}")
        return None


def extract_text_from_pdf(file_path: str) -> str | None:
    try:
        doc = fitz.open(file_path)
        return "".join(page.get_text("text") for page in doc)
    except Exception as e:
        print(f"  ❌ Błąd PDF {file_path}: {e}")
        return None


def process_all_books():
    if not os.path.exists(BOOKS_PATH):
        print(f"❌ Folder {BOOKS_PATH} nie istnieje!")
        return

    mode = "Claude API" if ANTHROPIC_API_KEY else "ekstrakcja lokalna"
    print(f"🔧 Tryb generowania outputu: {mode}\n")

    all_entries = []
    processed_count = 0

    for filename in sorted(os.listdir(BOOKS_PATH)):
        file_path = os.path.join(BOOKS_PATH, filename)

        if filename.lower().endswith(".pdf"):
            print(f"📖 PDF: {filename}")
            raw_text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith(".epub"):
            print(f"📚 EPUB: {filename}")
            raw_text = extract_text_from_epub(file_path)
        else:
            continue

        if not raw_text or len(raw_text.strip()) < 200:
            print(f"  ⚠️  Pominięto (mało treści): {filename}")
            continue

        cleaned = clean_text(raw_text)
        chunks = smart_chunking(cleaned, CHUNK_SIZE)

        for chunk in chunks:
            if len(chunk) < 100:  # pomijamy zbyt krótkie fragmenty
                continue
            instr = random.choice(INSTRUCTIONS)
            output = generate_output(instr, chunk)
            all_entries.append({
                "instruction": instr,
                "input": f"Źródło: {filename}\n\n{chunk}",
                "output": output,
            })

        processed_count += 1
        print(f"  ✅ {len(chunks)} chunków")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n✅ Przetworzono {processed_count} książek → {len(all_entries)} przykładów")
    print(f"📂 Dataset: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_all_books()
