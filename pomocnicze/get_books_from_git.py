import os
import requests

def download_pdfs_from_github(repo_url):
    # Przekształcenie URL repozytorium na URL API
    # Z: https://github.com/kelvin200/some-investment-books
    # Na: https://api.github.com/repos/kelvin200/some-investment-books/contents/
    
    api_url = repo_url.replace("github.com", "api.github.com/repos")
    if not api_url.endswith("/"):
        api_url += "/"
    api_url += "contents/"

    target_folder = "investment_books_pdf"
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"📁 Utworzono folder: {target_folder}")

    print(f"🔍 Łączenie z API: {api_url}")
    
    response = requests.get(api_url)
    
    if response.status_code == 200:
        contents = response.json()
        pdf_count = 0
        
        for item in contents:
            # Sprawdzamy czy element to plik i czy ma rozszerzenie .pdf
            if item['type'] == 'file' and item['name'].lower().endswith('.pdf'):
                file_url = item['download_url']
                file_name = item['name']
                
                print(f"📥 Pobieranie: {file_name}...")
                file_data = requests.get(file_url).content
                
                with open(os.path.join(target_folder, file_name), 'wb') as f:
                    f.write(file_data)
                pdf_count += 1
        
        print(f"\n✅ Gotowe! Pobrano {pdf_count} książek do folderu '{target_folder}'.")
    else:
        print(f"❌ Błąd podczas pobierania listy plików: {response.status_code}")
        print("Upewnij się, że URL jest poprawny lub spróbuj później (limity API GitHub).")

if __name__ == "__main__":
    REPO = "https://github.com/kelvin200/some-investment-books"
    download_pdfs_from_github(REPO)