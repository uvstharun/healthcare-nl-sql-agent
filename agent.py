import sqlite3
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

# -------------------------------------------------------
# DATABASE TOOL — same as before
# -------------------------------------------------------

def run_sql(query: str) -> str:
    try:
        conn = sqlite3.connect("medicare.db")
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


# -------------------------------------------------------
# RAG TOOL — new addition
# -------------------------------------------------------

def search_guidelines(query: str) -> str:
    chroma_client = chromadb.PersistentClient(path="docs/chroma_db")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.get_collection(
        name="clinical_guidelines",
        embedding_function=embedding_fn
    )
    results = collection.query(query_texts=[query], n_results=3)

    output = ""
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        output += f"\n[Source: {metadata['source']}]\n{doc}\n"
    return output


# -------------------------------------------------------
# TWO TOOLS — Claude decides which one(s) to use
# -------------------------------------------------------

tools = [
    {
        "name": "query_medicare_database",
        "description": """Query the CMS Medicare Part D prescriptions database using SQL.
        Use this tool for any question about prescription counts, drug costs, provider
        counts, geographic comparisons, or any question requiring numbers and statistics
        from Medicare data.

        Table: prescriptions
        Columns:
        - Prscrbr_NPI (TEXT): Unique provider identifier
        - Prscrbr_Last_Org_Name (TEXT): Provider last name or organization
        - Prscrbr_First_Name (TEXT): Provider first name
        - Prscrbr_City (TEXT): City where provider practices
        - Prscrbr_State_Abrvtn (TEXT): 2-letter state abbreviation (e.g. 'CA', 'TX')
        - Prscrbr_Type (TEXT): Provider specialty (e.g. 'Cardiologist', 'Family Practice')
        - Brnd_Name (TEXT): Brand name of the drug
        - Gnrc_Name (TEXT): Generic name of the drug
        - Tot_Clms (INTEGER): Total Medicare Part D claims
        - Tot_30day_Fills (REAL): Total 30-day equivalent fills
        - Tot_Day_Suply (INTEGER): Total days of drug supply
        - Tot_Drug_Cst (REAL): Total drug cost in US dollars
        - Tot_Benes (REAL): Total unique Medicare beneficiaries

        Notes:
        - NULL values = data suppressed by CMS for privacy
        - State always stored as 2-letter abbreviation
        - 26.7 million rows covering all US Medicare providers for 2023""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A valid SQLite SQL query against the prescriptions table."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_clinical_guidelines",
        "description": """Search clinical guidelines documents for evidence-based medical
        information. Use this tool for any question about treatment recommendations,
        clinical evidence, drug mechanisms, side effects, patient management guidelines,
        or any question requiring medical knowledge rather than statistics.

        Available documents:
        - Dietary advice and management for individuals with diabetes (ADA 2024)
        - Role of lipids and lipoproteins in atherosclerosis and statin therapy

        Use this tool when the question asks 'what do guidelines say', 'how should
        patients be treated', 'what is the evidence for', or similar clinical questions.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A plain English search query to find relevant clinical guideline content."
                }
            },
            "required": ["query"]
        }
    }
]


# -------------------------------------------------------
# AGENT LOOP — now handles multiple tool calls
# -------------------------------------------------------

def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "query_medicare_database":
        print(f"\n  [SQL] {tool_input['query']}")
        return run_sql(tool_input["query"])
    elif tool_name == "search_clinical_guidelines":
        print(f"\n  [RAG] Searching for: {tool_input['query']}")
        return search_guidelines(tool_input["query"])
    return "Unknown tool."


def ask_agent(user_question: str):
    print(f"\n{'='*80}")
    print(f"Question: {user_question}")
    print('='*80)

    messages = [{"role": "user", "content": user_question}]

    # Allow up to 5 tool calls per question for multi-tool questions
    for _ in range(5):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            tools=tools,
            messages=messages
        )

        # If no more tool calls needed — print final answer
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nAnswer: {block.text}")
            break

        # Process all tool calls in this response
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})


# -------------------------------------------------------
# TEST — pure SQL, pure RAG, and combined questions
# -------------------------------------------------------

if __name__ == "__main__":
    # Pure SQL question
    ask_agent("What are the top 5 states by total Medicare drug cost?")

    # Pure RAG question
    ask_agent("What do clinical guidelines say about dietary management for diabetes patients?")

    # Combined question — needs both tools
    ask_agent("How many Medicare providers prescribed Metformin in 2023, and what do guidelines say about its role in diabetes management?")