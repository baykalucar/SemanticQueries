import sqlite3
import pandas as pd
import os

def run_sql_query(query):
    db_conn = os.getenv("DB_CONNECTION_STRING")
    conn = sqlite3.connect(db_conn)
    query = query.replace("\\_", "_")
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df