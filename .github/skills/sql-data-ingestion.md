---
name: sql-data-ingestion
description: Pull data from SQL Server in chunks and save as a single merged parquet file
---

## Connection

```python
from sqlalchemy import create_engine, text

SERVER      = "DUMMY-SQL-SRV01"
DATABASE    = "DummyDatabase"
TABLE       = "dbo.dummy_table"
CHUNK_SIZE  = 50_000
OUTPUT_PATH = "data/data.parquet"

conn_str = (
    f"mssql+pyodbc://{SERVER}{DATABASE}"
    "?TrustServerCertificate=yes&driver=ODBC Driver 18 for SQL Server"
)
engine = create_engine(conn_str, fast_executemany=True)
```

## Chunked Load to Parquet

```python
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
chunks = []
with engine.connect() as conn:
    for i, chunk in enumerate(pd.read_sql(text(f"SELECT * FROM {TABLE}"), conn, chunksize=CHUNK_SIZE)):
        chunks.append(chunk)
        print(f"  chunk {i+1}: {len(chunk):,} rows loaded")
df = pd.concat(chunks, ignore_index=True)
print(f"Total rows: {len(df):,} | Columns: {df.shape[1]}")
df.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved -> {OUTPUT_PATH}")
```

## Notes
- Use chunksize to avoid OOM on large tables.
- TrustServerCertificate=yes required for internal servers.
- Output is always a single merged parquet at data/data.parquet.
