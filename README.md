# Healthcare AI Agent — NL-to-SQL + Clinical Guidelines

A multi-tool AI agent that answers healthcare questions in plain English by querying Medicare prescribing data and searching clinical guidelines simultaneously.

Built with the Anthropic Claude API using raw tool-calling — no LangChain, no LangGraph.

---

## Demo

> Add your demo video link here after uploading

---

## What It Does

Ask a question in plain English. The agent decides which tool to use — or uses both:

| Question type | Tool used |
|---|---|
| "How many providers prescribed Metformin in California?" | SQL only |
| "What do guidelines say about diabetes dietary management?" | RAG only |
| "How many providers prescribed Metformin and what do guidelines say about it?" | Both tools |

The agent writes its own SQL, searches clinical PDFs, and synthesizes both into one answer — without you writing a single line of SQL.

---

## Architecture

```
User Question
      ↓
HTML Frontend → FastAPI (api.py) → Agent Loop
                                        ↓
                              Claude (claude-haiku-4-5)
                              decides which tool(s) to call
                                   ↓            ↓
                              SQL Tool       RAG Tool
                              SQLite DB      ChromaDB
                              CMS Medicare   Clinical PDFs
                                   ↓            ↓
                              Results returned to Claude
                                        ↓
                              Plain English answer
```

**Key design decision:** No frameworks. The tool-calling loop is written from scratch so every step is transparent and debuggable. The schema description in the tool definition is treated as code — precise column descriptions directly control SQL accuracy.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Anthropic Claude (claude-haiku-4-5) |
| Agent pattern | Raw tool-calling via Anthropic SDK |
| API layer | FastAPI + uvicorn |
| Structured data | SQLite — CMS Medicare Part D 2023 |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Document parsing | pypdf |
| Data loading | pandas (chunked) |
| Frontend | Custom HTML/CSS/JS |
| Streamlit UI | app.py (alternative interface) |
| Language | Python 3.13 |

No LangChain. No LangGraph. No fine-tuning.

---

## Sample Questions

**SQL query:**
```
Q: How many providers prescribed Metformin in California?

Generated SQL:
SELECT COUNT(DISTINCT Prscrbr_NPI) as unique_providers
FROM prescriptions
WHERE Gnrc_Name LIKE '%metformin%'
AND Prscrbr_State_Abrvtn = 'CA'

Answer: 18,432 providers in California prescribed Metformin in 2023.
```

**RAG query:**
```
Q: What do guidelines say about diabetes dietary management?

Answer: Based on ADA 2024 guidelines, dietary management should be
individualized — no one-size-fits-all approach. Medical Nutrition
Therapy (MNT) by a Registered Dietitian is recommended at diagnosis.
MNT reduces A1C by 0.3-2.0% in type 2 diabetes.
```

**Combined query:**
```
Q: How many providers prescribed Metformin and what do guidelines say about it?

SQL Tool  → 255,046 Medicare providers prescribed Metformin in 2023
RAG Tool  → ADA guidelines emphasize comprehensive diabetes management
            combining MNT, DSMES, and pharmacological treatment

Answer: Both results synthesized into one plain English response.
```

---

## How to Run It

### 1. Clone the repo
```bash
git clone https://github.com/uvstharun/healthcare-nl-sql-agent.git
cd healthcare-nl-sql-agent
```

### 2. Set up environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your Anthropic API key
```bash
touch .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
```

### 4. Download the CMS data
Download the Medicare Part D Prescribers by Provider and Drug dataset (2023):
https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider-and-drug

Place the CSV file in the data/ folder.

### 5. Load data into SQLite
```bash
python load_cms_data.py
```

### 6. Download clinical guidelines PDFs
```bash
python download_docs.py
```
Or manually save any clinical PDF into the docs/ folder.

### 7. Build the vector store
```bash
python build_vectorstore.py
```

### 8. Start the API
```bash
uvicorn api:app --reload --port 8000
```

### 9. Open the frontend
```bash
open healthcare_agent_ui.html
```

Or run the Streamlit app instead:
```bash
streamlit run app.py
```

---

## Project Structure

```
healthcare-nl-sql-agent/
<<<<<<< HEAD
├── agent.py                  # Core agent — tool-calling loop
├── api.py                    # FastAPI backend — serves HTML frontend
├── app.py                    # Streamlit UI (alternative interface)
├── build_vectorstore.py      # Builds ChromaDB from clinical PDFs
├── download_docs.py          # Downloads clinical guideline PDFs
├── explore_data.py           # Data exploration script
├── load_cms_data.py          # Loads CMS CSV into SQLite in chunks
├── setup_db.py               # Fake data setup for initial testing
├── test_connection.py        # Anthropic API connection test
├── test_rag.py               # RAG retrieval test
├── healthcare_agent_ui.html  # HTML frontend connected to API
├── requirements.txt
├── data/                     # CMS CSV goes here (not tracked in git)
├── docs/                     # Clinical PDFs + ChromaDB (not tracked)
├── .env                      # API key (not tracked in git)
└── .gitignore
=======

https://github.com/user-attachments/assets/e15d9342-8aac-4eb6-b935-3d42d47e3758


├── agent.py              # Main agent — tool-calling loop
├── load_cms_data.py      # Loads CMS CSV into SQLite in chunks
├── setup_db.py           # Fake data setup for initial testing
├── explore_data.py       # Data exploration script
├── test_connection.py    # API connection test
├── data/                 # CMS CSV goes here (not tracked in git)
├── .env                  # API key (not tracked in git)
├── .gitignore
└── README.md
>>>>>>> ed1291732f4a5a618228ff81d1d8c149eafe3000
```

---
## Demo




https://github.com/user-attachments/assets/7f1ebe63-5054-442f-aa99-2eb62ef40613







---

## Why This Project

Most NL-to-SQL demos use toy datasets. Most RAG demos use simple Q&A over a single PDF.

This agent combines both patterns against real healthcare data — the same kind of data that exists inside every health system in the US. The domain complexity (provider specialties, drug formularies, suppressed CMS values, clinical coding) makes this a realistic healthcare AI engineering problem, not a tutorial exercise.

The goal: demonstrate that healthcare domain knowledge combined with LLM tool-calling can make complex government datasets and clinical guidelines accessible to non-technical clinical and operations staff.

---



## Author

**Vishnu Sai** — Data Scientist | Healthcare AI
[LinkedIn](https://www.linkedin.com/in/vishnusai29/) · [GitHub](https://github.com/uvstharun)
