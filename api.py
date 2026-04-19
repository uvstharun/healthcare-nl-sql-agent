from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Healthcare AI Agent API")

# Allow the HTML file to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# LOAD MODELS ONCE AT STARTUP
# -------------------------------------------------------

client = anthropic.Anthropic()

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.PersistentClient(path="docs/chroma_db")
collection = chroma_client.get_collection(
    name="clinical_guidelines",
    embedding_function=embedding_fn
)

print("Models loaded. Pre-warming embedding model...")
collection.query(query_texts=["test"], n_results=1)
print("Embedding model warmed. API ready.")

# -------------------------------------------------------
# TOOL FUNCTIONS
# -------------------------------------------------------

def run_sql(query: str) -> str:
    try:
        conn = sqlite3.connect("medicare.db")
        conn.execute("PRAGMA cache_size = -64000")
        conn.execute("PRAGMA temp_store = MEMORY")
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()

        if not rows:
            return "No results found."

        result = " | ".join(columns) + "\n"
        result += "-" * 80 + "\n"
        for row in rows:
            result += " | ".join(str(val) for val in row) + "\n"
        return result

    except Exception as e:
        return f"SQL Error: {str(e)}"


def search_guidelines(query: str) -> str:
    results = collection.query(query_texts=[query], n_results=3)
    output = ""
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        output += f"\n[Source: {metadata['source']}]\n{doc}\n"
    return output


# -------------------------------------------------------
# TOOL DEFINITIONS
# -------------------------------------------------------

tools = [
    {
        "name": "query_medicare_database",
        "description": """Query the CMS Medicare Part D prescriptions database using SQL.
        Use for questions about prescription counts, drug costs, provider counts,
        geographic comparisons, or any question requiring numbers and statistics.

        Table: prescriptions
        Columns:
        - Prscrbr_NPI (TEXT): Unique provider identifier
        - Prscrbr_Last_Org_Name (TEXT): Provider last name or organization
        - Prscrbr_First_Name (TEXT): Provider first name
        - Prscrbr_City (TEXT): City where provider practices
        - Prscrbr_State_Abrvtn (TEXT): 2-letter state abbreviation (e.g. 'CA', 'TX')
        - Prscrbr_Type (TEXT): Provider specialty
        - Brnd_Name (TEXT): Brand name of the drug
        - Gnrc_Name (TEXT): Generic name of the drug
        - Tot_Clms (INTEGER): Total Medicare Part D claims
        - Tot_30day_Fills (REAL): Total 30-day equivalent fills
        - Tot_Day_Suply (INTEGER): Total days of drug supply
        - Tot_Drug_Cst (REAL): Total drug cost in US dollars
        - Tot_Benes (REAL): Total unique Medicare beneficiaries
        State is always 2-letter abbreviation. NULL = CMS suppressed for privacy.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A valid SQLite SQL query."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_clinical_guidelines",
        "description": """Search clinical guidelines for evidence-based medical information.
        Use for questions about treatment recommendations, clinical evidence, drug
        mechanisms, patient management, or any question needing medical knowledge.

        Available documents:
        - Dietary advice for individuals with diabetes (ADA 2024)
        - Role of lipids and lipoproteins in atherosclerosis and statin therapy""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Plain English search query."}
            },
            "required": ["query"]
        }
    }
]


# -------------------------------------------------------
# AGENT LOOP
# -------------------------------------------------------

def run_agent(question: str):
    messages = [{"role": "user", "content": question}]
    tool_calls_made = []

    for _ in range(5):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            answer = ""
            for block in response.content:
                if hasattr(block, "text"):
                    answer = block.text
            return answer, tool_calls_made

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    if block.name == "query_medicare_database":
                        result = run_sql(block.input["query"])
                        tool_calls_made.append({
                            "type": "SQL",
                            "input": block.input["query"],
                            "result": result
                        })
                    elif block.name == "search_clinical_guidelines":
                        result = search_guidelines(block.input["query"])
                        tool_calls_made.append({
                            "type": "RAG",
                            "input": block.input["query"],
                            "result": result
                        })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})

    return "Agent could not complete the request.", tool_calls_made


# -------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str

# Simple in-memory cache
query_cache = {}

@app.get("/")
def root():
    return {"status": "Healthcare AI Agent API is running"}

@app.post("/ask")
def ask(request: QuestionRequest):
    # Return cached result if same question asked before
    if request.question in query_cache:
        print(f"Cache hit: {request.question}")
        return query_cache[request.question]

    answer, tool_calls = run_agent(request.question)
    result = {"answer": answer, "tool_calls": tool_calls}

    # Store in cache
    query_cache[request.question] = result
    return result