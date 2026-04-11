# Healthcare NL-to-SQL Agent

A conversational AI agent that translates plain English questions into SQL queries against real Medicare Part D data — no SQL knowledge required.

Built with the Anthropic Claude API using raw tool-calling (no LangChain), running against 26.7 million rows of CMS Medicare prescribing data.

---

## What It Does

Instead of writing SQL, a user can ask:

> *"Which state has the highest total drug cost?"*

The agent:
1. Sends the question to Claude along with the database schema
2. Claude generates a SQL query using tool-calling
3. The query runs against the local SQLite database (26.7M rows)
4. Results are returned to Claude
5. Claude explains the answer in plain English

---

## Architecture

```
User Question (plain English)
        ↓
Claude reads schema → writes SQL via tool_use
        ↓
Python executes SQL against SQLite (26.7M rows)
        ↓
Results returned to Claude
        ↓
Claude narrates answer in plain English
```

**Key design decision:** The tool description (schema) is treated as code.
Precise column descriptions directly control SQL accuracy — no fine-tuning needed.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Anthropic Claude (claude-haiku-4-5) |
| Agent pattern | Raw tool-calling via Anthropic SDK |
| Database | SQLite |
| Data loading | pandas (chunked, 100k rows at a time) |
| Data source | CMS Medicare Part D Prescribers 2023 |
| Language | Python 3.13 |

No LangChain. No LangGraph. No vector database. Just the Anthropic SDK and SQLite.

---

## Sample Questions and Answers

**Q: Which state has the highest total drug cost?**
```
Generated SQL:
SELECT Prscrbr_State_Abrvtn, SUM(Tot_Drug_Cst) as Total_Drug_Cost
FROM prescriptions
WHERE Prscrbr_State_Abrvtn IS NOT NULL
GROUP BY Prscrbr_State_Abrvtn
ORDER BY Total_Drug_Cost DESC LIMIT 1

Answer: California (CA) has the highest total drug cost at $20.9 billion
in Medicare Part D prescription drug costs for 2023.
```

**Q: What are the top 5 most prescribed generic drugs by total claims?**
```
Answer:
1. Atorvastatin Calcium — 67,633,912 claims
2. Amlodipine Besylate — 46,601,898 claims
3. Levothyroxine Sodium — 45,138,029 claims
4. Lisinopril — 35,257,305 claims
5. Gabapentin — 33,913,654 claims
```

**Q: How many unique providers are there in California?**
```
Answer: There are 110,430 unique providers in California
in the Medicare Part D prescriptions database.
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
pip install anthropic python-dotenv pandas
```

### 3. Add your Anthropic API key
```bash
touch .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
```

### 4. Download the CMS data
Download the Medicare Part D Prescribers by Provider and Drug dataset (2023) from:
https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider-and-drug

Place the CSV file in the `data/` folder.

### 5. Load data into SQLite
```bash
python load_cms_data.py
```
This loads 26.7 million rows in chunks — takes 2-4 minutes.

### 6. Run the agent
```bash
python agent.py
```

---

## Project Structure

```
healthcare-nl-sql-agent/
├── agent.py              # Main agent — tool-calling loop
├── load_cms_data.py      # Loads CMS CSV into SQLite in chunks
├── setup_db.py           # Fake data setup for initial testing
├── explore_data.py       # Data exploration script
├── test_connection.py    # API connection test
├── data/                 # CMS CSV goes here (not tracked in git)
├── .env                  # API key (not tracked in git)
├── .gitignore
└── README.md
```

---

## Why This Project

Most NL-to-SQL demos use toy datasets with 5 columns and 100 rows.

This agent runs against **26.7 million real Medicare records** — the kind of data that exists inside every health system in the US. The domain complexity (provider specialties, drug formularies, suppressed CMS values, beneficiary counts) makes this a realistic healthcare AI engineering problem, not a tutorial exercise.

The goal is to demonstrate that healthcare domain knowledge combined with LLM tool-calling can make complex government datasets accessible to non-technical clinical and operations staff.

---

## Author

**Vishnu Sai** — Data Scientist | Healthcare AI
[LinkedIn](https://www.linkedin.com/in/vishnusai29/) · [GitHub](https://github.com/uvstharun)
