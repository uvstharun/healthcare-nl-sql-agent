import sqlite3
import pandas as pd

# Create a tiny fake Medicare-style dataset
data = {
    "provider_name": ["Dr. Smith", "Dr. Jones", "Dr. Patel", "Dr. Lee", "Dr. Garcia"],
    "drug_name": ["Lisinopril", "Metformin", "Atorvastatin", "Amlodipine", "Metformin"],
    "total_claims": [1200, 850, 2100, 430, 1750],
    "total_patients": [300, 210, 520, 110, 440],
    "total_cost": [14500.00, 9200.00, 87000.00, 5100.00, 19800.00],
    "state": ["CA", "TX", "CA", "NY", "FL"]
}

df = pd.DataFrame(data)

# Save to SQLite database
conn = sqlite3.connect("medicare.db")
df.to_sql("prescriptions", conn, if_exists="replace", index=False)
conn.close()

print("Database created successfully.")
print(df)