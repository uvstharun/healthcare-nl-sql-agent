import pandas as pd
import sqlite3
import os

filepath = "data/MUP_DPR_RY25_P04_V10_DY23_NPIBN.csv"
db_path = "medicare.db"

# Columns we actually need — ignore the rest to save memory
COLS_TO_KEEP = [
    "Prscrbr_NPI",
    "Prscrbr_Last_Org_Name",
    "Prscrbr_First_Name",
    "Prscrbr_City",
    "Prscrbr_State_Abrvtn",
    "Prscrbr_Type",
    "Brnd_Name",
    "Gnrc_Name",
    "Tot_Clms",
    "Tot_30day_Fills",
    "Tot_Day_Suply",
    "Tot_Drug_Cst",
    "Tot_Benes"
]

# Delete old database if it exists
if os.path.exists(db_path):
    os.remove(db_path)
    print("Removed old medicare.db")

conn = sqlite3.connect(db_path)

chunk_size = 100_000
total_rows = 0
first_chunk = True

print("Loading CMS data into SQLite...")
print("This will take 2-4 minutes. Grab a coffee.\n")

for chunk in pd.read_csv(
    filepath,
    usecols=COLS_TO_KEEP,
    chunksize=chunk_size,
    low_memory=False
):
    # Replace CMS suppression flags with None
    chunk = chunk.replace({"#": None, "*": None})

    # Write to SQLite
    chunk.to_sql(
        "prescriptions",
        conn,
        if_exists="append" if not first_chunk else "replace",
        index=False
    )

    total_rows += len(chunk)
    first_chunk = False
    print(f"  Loaded {total_rows:,} rows...")

conn.close()
print(f"\nDone. Total rows loaded: {total_rows:,}")
print(f"Database saved to: {db_path}")