import pandas as pd
import sqlite3

# Load CSV
csv_file = "data/quick_commerce_orders_gold_20260422.csv"

# Create SQLite database
db_file = "data.sqlite"

# Read CSV
df = pd.read_csv(csv_file)

# Connect to SQLite
conn = sqlite3.connect(db_file)

# Write table to SQLite
df.to_sql("quick_commerce_orders_gold", conn, if_exists="replace", index=False)

# Close connection
conn.close()

print("CSV successfully converted to SQLite database!")