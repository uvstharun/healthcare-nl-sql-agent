import pandas as pd

filepath = "data/MUP_DPR_RY25_P04_V10_DY23_NPIBN.csv"

# Read only the first 5 rows — don't load the whole 3.88 GB
df = pd.read_csv(filepath, nrows=5, low_memory=False)

print("=== COLUMNS ===")
for col in df.columns:
    print(f"  {col}")

print(f"\n=== SHAPE (first 5 rows) ===")
print(f"  Columns: {len(df.columns)}")

print(f"\n=== SAMPLE DATA ===")
print(df.head(2).to_string())