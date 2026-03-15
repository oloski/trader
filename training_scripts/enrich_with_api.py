import os
import json
import time
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv

# --- KONFIGURACJA ---
load_dotenv() # Wczytuje .env z folderu
API_KEY = os.getenv("GEMINI_API_KEY")
INPUT_FILE = "/app/shared_data/books_training.jsonl"
OUTPUT_FILE = "/app/shared_data/books_training_final.jsonl"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_expert_analysis(instruction, context):
    """Wysyła zapytanie do API o profesjonalną analizę."""
    prompt = f"""
    Jesteś ekspertem od handlu algorytmicznego i psychologii rynkowej. 
    Twoim zadaniem jest stworzenie wysokiej jakości odpowiedzi na poniższą instrukcję, 
    bazując na dostarczonym fragmencie książki.
    
    ZADANIE: {instruction}
    FRAGMENT TEKSTU:
    {context}
    
    ODPOWIEDŹ (konkretna, techniczna, bez zbędnych wstępów):
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Błąd API: {e}. Czekam 5s...")
        time.sleep(5)
        return None

def enrich_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Nie znaleziono pliku {INPUT_FILE}. Najpierw uruchom preprocess_books.py!")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    final_data = []
    print(f"🚀 Wzbogacanie {len(lines)} przykładów przez API Gemini...")

    for line in tqdm(lines):
        entry = json.loads(line)
        
        # Pobieramy analizę (z retry logic)
        analysis = None
        while analysis is None:
            analysis = get_expert_analysis(entry['instruction'], entry['input'])
        
        entry['output'] = analysis
        final_data.append(entry)

        # Zapisujemy na bieżąco, żeby nie stracić postępu przy błędzie sieci
        if len(final_data) % 10 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
                for item in final_data:
                    out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Limitowanie tempa (Free tier ma limity RPM - Requests Per Minute)
        time.sleep(1) 

    print(f"✅ Sukces! Dataset wzbogacony: {OUTPUT_FILE}")

if __name__ == "__main__":
    enrich_dataset()