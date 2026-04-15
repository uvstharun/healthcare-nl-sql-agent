import streamlit as st
import sqlite3
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Healthcare AI Agent",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare NL-to-SQL Agent")
st.caption("Ask questions about Medicare prescribing data and clinical guidelines in plain English.")

# -------------------------------------------------------
# LOAD MODELS ONCE — each cached separately so nothing
# reloads between queries
# -------------------------------------------------------

@st.cache_resource
def load_anthropic_client():
    return anthropic.Anthropic()

@st.cache_resource
def load_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_chroma_collection():
    chroma_client = chromadb.PersistentClient(path="docs/chroma_db")
    embedding_fn = load_embedding_function()
    return chroma_client.get_collection(
        name="clinical_guidelines",
        embedding_function=embedding_fn
    )

client = load_anthropic_client()
collection = load_chroma_collection()

# -------------------------------------------------------
# TOOL FUNCTIONS
# -------------------------------------------------------

def run_sql(query: str) -> str:
    try:
        conn = sqlite3.connect("medicare.db")
        conn.execute("PRAGMA cache_size = -64000")   # 64MB cache
        conn.execute("PRAGMA temp_store = MEMORY")   # temp tables in RAM
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

def run_agent(user_question: str):
    messages = [{"role": "user", "content": user_question}]
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
# STREAMLIT UI
# -------------------------------------------------------

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

# Sidebar
with st.sidebar:
    st.header("💡 Example Questions")

    st.markdown("**📊 Data questions**")
    data_questions = [
        "How many providers prescribed Metformin in California?",
        "Which specialty prescribes the most Atorvastatin?",
        "What are the top 5 most prescribed generic drugs?",
        "How many unique providers are there in Texas?",
        "What is the average drug cost per claim in New York?",
    ]
    for q in data_questions:
        if st.button(q, key=f"dq_{q}", use_container_width=True):
            st.session_state.selected_question = q

    st.markdown("**📋 Clinical questions**")
    clinical_questions = [
        "What do guidelines say about diabetes dietary management?",
        "What is the evidence for statin therapy in heart disease?",
    ]
    for q in clinical_questions:
        if st.button(q, key=f"cq_{q}", use_container_width=True):
            st.session_state.selected_question = q

    st.markdown("**🔀 Combined questions**")
    combined_questions = [
        "How many providers prescribed Metformin and what do guidelines say about it?",
        "How many claims for Atorvastatin and what do guidelines say about statins?",
    ]
    for q in combined_questions:
        if st.button(q, key=f"comb_{q}", use_container_width=True):
            st.session_state.selected_question = q

    st.divider()
    st.caption("🔵 SQL = Medicare database query")
    st.caption("🟢 RAG = Clinical guidelines search")
    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "tool_calls" in msg and msg["tool_calls"]:
            for call in msg["tool_calls"]:
                icon = "🔵" if call["type"] == "SQL" else "🟢"
                label = f"{icon} {call['type']} — {call['input'][:60]}..."
                with st.expander(label):
                    if call["type"] == "SQL":
                        st.code(call["input"], language="sql")
                    else:
                        st.caption(f"Search: {call['input']}")
                    preview = call["result"][:800] + "..." if len(call["result"]) > 800 else call["result"]
                    st.text(preview)

# Chat input
question = st.chat_input("Ask about Medicare data or clinical guidelines...")

# Handle sidebar button clicks
if st.session_state.selected_question and not question:
    question = st.session_state.selected_question
    st.session_state.selected_question = ""

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, tool_calls = run_agent(question)

        if tool_calls:
            tool_types = list(set(call["type"] for call in tool_calls))
            badge_text = " + ".join(
                f"{'🔵 SQL' if t == 'SQL' else '🟢 RAG'}" for t in tool_types
            )
            st.caption(f"Tools used: {badge_text}")

        st.markdown(answer)

        for call in tool_calls:
            icon = "🔵" if call["type"] == "SQL" else "🟢"
            label = f"{icon} {call['type']} — {call['input'][:60]}..."
            with st.expander(label):
                if call["type"] == "SQL":
                    st.code(call["input"], language="sql")
                else:
                    st.caption(f"Search: {call['input']}")
                preview = call["result"][:800] + "..." if len(call["result"]) > 800 else call["result"]
                st.text(preview)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool_calls": tool_calls
    })