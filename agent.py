import sqlite3
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

def run_sql(query: str) -> str:
    """Execute a SQL query and return results as a string."""
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


tools = [
    {
        "name": "run_sql_query",
        "description": """Execute a SQL query against the CMS Medicare Part D prescriptions database.
        The database has one table called 'prescriptions' with these columns:

        - Prscrbr_NPI (TEXT): Unique provider identifier (National Provider Identifier)
        - Prscrbr_Last_Org_Name (TEXT): Provider last name or organization name
        - Prscrbr_First_Name (TEXT): Provider first name
        - Prscrbr_City (TEXT): City where the provider practices
        - Prscrbr_State_Abrvtn (TEXT): 2-letter US state abbreviation (e.g. 'CA', 'TX', 'NY')
        - Prscrbr_Type (TEXT): Provider specialty (e.g. 'Hospitalist', 'Cardiologist', 'Family Practice')
        - Brnd_Name (TEXT): Brand name of the drug prescribed
        - Gnrc_Name (TEXT): Generic name of the drug prescribed
        - Tot_Clms (INTEGER): Total number of Medicare Part D claims filed
        - Tot_30day_Fills (REAL): Total number of 30-day equivalent fills
        - Tot_Day_Suply (INTEGER): Total days of drug supply dispensed
        - Tot_Drug_Cst (REAL): Total drug cost in US dollars
        - Tot_Benes (REAL): Total number of unique Medicare beneficiaries (patients)

        Important notes:
        - NULL values mean data was suppressed by CMS for patient privacy
        - Always use SUM(), AVG(), COUNT() for aggregations across providers or drugs
        - State is always stored as 2-letter abbreviation
        - Drug names may appear in both brand and generic forms — use Gnrc_Name for consistency
        - The dataset covers Medicare Part D (prescription drug) claims for 2023
        - There are 26.7 million rows covering all US Medicare providers and drugs

        Use this tool for every question about prescriptions, providers, drugs, costs, or patients.""",
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
    }
]


def ask_agent(user_question: str):
    print(f"\nQuestion: {user_question}")
    print("-" * 80)

    messages = [{"role": "user", "content": user_question}]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    if response.stop_reason == "tool_use":
        tool_use_block = next(
            block for block in response.content
            if block.type == "tool_use"
        )

        sql_query = tool_use_block.input["query"]
        print(f"Generated SQL:\n{sql_query}\n")

        sql_result = run_sql(sql_query)
        print(f"Query Results:\n{sql_result}")

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": sql_result
                }
            ]
        })

        final_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        print(f"\nAnswer: {final_response.content[0].text}")

    else:
        print(f"Answer: {response.content[0].text}")


if __name__ == "__main__":
    ask_agent("Which state has the highest total drug cost?")
    ask_agent("What are the top 5 most prescribed generic drugs by total claims?")
    ask_agent("How many unique providers are there in California?")