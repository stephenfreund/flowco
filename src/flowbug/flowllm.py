import argparse
import sqlite3
import json

import streamlit as st

# ——— parse out our own args, ignore Streamlit’s flags ———
parser = argparse.ArgumentParser()
parser.add_argument(
    "db",
    help="Path to the SQLite responses database file",
)
args, _ = parser.parse_known_args()
DB_PATH = args.db

@st.cache_data
def get_ids():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT id FROM responses ORDER BY id")
    ids = [row[0] for row in cur.fetchall()]
    conn.close()
    return ids

@st.cache_data
def get_records_for_id(selected_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, model, response_model, messages
        FROM responses
        WHERE id = ?
        ORDER BY timestamp
    """, (selected_id,))
    rows = cur.fetchall()
    conn.close()

    records = []
    for ts, model, resp_schema_json, messages_json in rows:
        resp_schema = json.loads(resp_schema_json) if resp_schema_json else None
        messages    = json.loads(messages_json)    if messages_json    else None
        records.append({
            "timestamp":      ts,
            "model":          model,
            "response_model": resp_schema,
            "messages":       messages,
        })
    return records

st.title("Flowco Responses Browser")

ids = get_ids()
if not ids:
    st.warning(f"No records found in `{DB_PATH}`.")
else:
    selected_id = st.selectbox("Choose an assistant ID", ids)
    st.markdown(f"### Records for ID: `{selected_id}`")
    records = get_records_for_id(selected_id)
    if records:
        st.json(records)
    else:
        st.info("No records for this ID.")
