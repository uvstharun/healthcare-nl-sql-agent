import requests
import os

# Create docs folder if it doesn't exist
os.makedirs("docs", exist_ok=True)

# Clinical guidelines directly relevant to top drugs in our Medicare data
pdfs = {
    "metformin_diabetes": "https://www.ncbi.nlm.nih.gov/books/NBK279012/pdf/Bookshelf_NBK279012.pdf",
    "statin_guidelines": "https://www.ncbi.nlm.nih.gov/books/NBK343489/pdf/Bookshelf_NBK343489.pdf",
}
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

for name, url in pdfs.items():
    print(f"Downloading {name}...")
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            filepath = f"docs/{name}.pdf"
            with open(filepath, "wb") as f:
                f.write(response.content)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  Saved {filepath} ({size_kb:.0f} KB)")
        else:
            print(f"  Failed — status code {response.status_code}")
    except Exception as e:
        print(f"  Error: {str(e)}")

print("\nDone. Files in docs/:")
for f in os.listdir("docs"):
    print(f"  {f}")