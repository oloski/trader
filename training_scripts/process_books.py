import fitz  # PyMuPDF
import os
import json
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

BOOKS_PATH = "/app/shared_data/library/books"
OUTPUT_FILE = "/app/shared_data/books_training.jsonl"

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

def extract_text_from_epub(file_path):
    try:
        book = epub.read_epub(file_path)
        text_content = ""
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                text_content += clean_html(item.get_content()) + "\n"
        return text_content
    except Exception as e:
        print(f" ❌ Błąd krytyczny EPUB {file_path}: {e}")
        return None

def extract_text_from_pdf(file_path):
    try:
        # Próbujemy otworzyć PDF
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f" ❌ Pomijam PDF (prawdopodobnie uszkodzony): {file_path}")
        print(f"    Szczegóły błędu: {e}")
        return None

def process_all_books():
    all_entries = []
    if not os.path.exists(BOOKS_PATH):
        print(f"❌ Folder {BOOKS_PATH} nie istnieje!")
        return

    processed_count = 0
    failed_count = 0

    for filename in os.listdir(BOOKS_PATH):
        file_path = os.path.join(BOOKS_PATH, filename)
        text = ""
        
        if filename.lower().endswith(".pdf"):
            print(f"📖 Przetwarzanie PDF: {filename}...")
            text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith(".epub"):
            print(f"📚 Przetwarzanie EPUB: {filename}...")
            text = extract_text_from_epub(file_path)
        else:
            continue

        if text and len(text.strip()) > 100:
            chunks = [text[i:i+2500] for i in range(0, len(text), 2500)]
            for chunk in chunks:
                all_entries.append({
                    "instruction": f"Wyjaśnij koncepcję handlową zawartą w książce: {filename}",
                    "input": chunk.strip(),
                    "output": "Zintegrowałem tę wiedzę z moją bazą strategii."
                })
            processed_count += 1
        else:
            print(f" ⚠️  Nie udało się wyciągnąć tekstu z: {filename}")
            failed_count += 1

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Raport końcowy:")
    print(f"   - Przetworzono pomyślnie: {processed_count}")
    print(f"   - Pominięto (błędy): {failed_count}")
    print(f"   - Plik wynikowy: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_books()