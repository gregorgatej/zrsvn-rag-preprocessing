
import os
import json
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from transformers import AutoTokenizer
import semchunk
import numpy as np

# ─── CONFIG & ADAPTERS ─────────────────────────────────────────────────────────
load_dotenv()
TOKENIZER_NAME = "BAAI/bge-m3"
SEMTEXT_CHUNK_SIZE = 512
PREP_THRESHOLD   = SEMTEXT_CHUNK_SIZE / 2
CHUNK_THRESHOLD  = SEMTEXT_CHUNK_SIZE * 1.5

DB_PARAMS = {
    "dbname":   "zrsvn",
    "user":     "ggatej-pg",
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host":     "localhost",
    "port":     "5432",
    "options":  "-c search_path=rag_najdbe"
}

# register numpy adapters
register_adapter(np.ndarray, lambda arr: AsIs(arr.tolist()))
register_adapter(np.float32,   lambda val: AsIs(val))
register_adapter(np.float64,   lambda val: AsIs(val))


def _get_conn():
    return psycopg2.connect(**DB_PARAMS)


# ─── STEP 1: JSON → files/sections/[…]
def insert_file_and_sections(json_path: str):
    conn = _get_conn(); cur = conn.cursor()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    cur.execute(
        "INSERT INTO files (type, filename, s3_key) VALUES (%s,%s,%s) RETURNING id",
        (data["fileType"], data["fileName"], data["fileS3Path"])
    )
    file_id = cur.fetchone()[0]

    section_map = {}
    for page in data["documentPages"]:
        page_nr = page["pageNumber"]
        for chunk in page["chunks"]:
            sid_json = chunk["sectionID"]
            if sid_json not in section_map:
                cur.execute(
                    "INSERT INTO sections (header, file_id) VALUES (%s,%s) RETURNING id",
                    (chunk["sectionHeader"], file_id)
                )
                section_map[sid_json] = cur.fetchone()[0]

            se_id = section_map[sid_json]
            bbox = chunk["boundingBox"][0]
            s3key = chunk.get("chunkLocalPath")

            cur.execute(
                """
                INSERT INTO section_elements
                  (type, file_seq_position, section_seq_position,
                   page_nr, bounding_box, s3_key, section_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id
                """,
                (
                    chunk["contentType"][0],
                    chunk["fileSeqPosition"],
                    chunk["sectionSeqPosition"],
                    page_nr,
                    json.dumps(bbox),
                    s3key,
                    se_id
                )
            )
            selem_id = cur.fetchone()[0]

            ctype = chunk["contentType"][0]
            if ctype == "paragraph":
                text = chunk["text"]
                nr_chars = chunk["nrCharacters"]
                nr_tokens = len(tokenizer.tokenize(text))
                cur.execute(
                    """
                    INSERT INTO paragraphs
                      (text, nr_tokens, nr_characters, section_element_id)
                    VALUES (%s,%s,%s,%s)
                    """,
                    (text, nr_tokens, nr_chars, selem_id)
                )
            elif ctype == "picture":
                cur.execute("INSERT INTO pictures (section_element_id) VALUES (%s)", (selem_id,))
            elif ctype == "table":
                cur.execute("INSERT INTO tables (section_element_id) VALUES (%s)", (selem_id,))

    conn.commit()
    cur.close(); conn.close()


# ─── STEP 2: GROUP & CHUNK PARAGRAPHS
def prepare_and_chunk_texts():
    conn = _get_conn(); cur = conn.cursor()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    chunker = semchunk.chunkerify(tokenizer, SEMTEXT_CHUNK_SIZE)

    # fetch all paragraphs in order
    cur.execute(
        """
        SELECT p.id, p.text, p.nr_tokens, se.section_id
        FROM paragraphs p
        JOIN section_elements se ON p.section_element_id = se.id
        ORDER BY se.section_id, se.section_seq_position
        """
    )
    rows = cur.fetchall()

    i = 0
    while i < len(rows):
        pid, text, tokens, sec_id = rows[i]
        group_text   = text
        group_tokens = tokens
        group_ids    = [pid]

        # if this paragraph is “small”, absorb siblings until threshold or new section
        if tokens < PREP_THRESHOLD:
            j = i + 1
            while (
                j < len(rows)
                and rows[j][3] == sec_id
                and group_tokens < PREP_THRESHOLD
            ):
                _, next_text, next_tokens, _ = rows[j]
                group_text   += " " + next_text
                group_tokens += next_tokens
                group_ids.append(rows[j][0])
                j += 1
            next_i = j
        else:
            next_i = i + 1

        # insert into prepared_texts
        cur.execute(
            """
            INSERT INTO prepared_texts
              (text, nr_tokens, text_chunk_size_threshold)
            VALUES (%s,%s,%s) RETURNING id
            """,
            (group_text, group_tokens, PREP_THRESHOLD)
        )
        pt_id = cur.fetchone()[0]

        # link back
        cur.execute(
            "UPDATE paragraphs SET prepared_text_id = %s WHERE id = ANY(%s)",
            (pt_id, group_ids)
        )
        i = next_i

    # now chunk each prepared_text
    cur.execute("SELECT id, text, nr_tokens FROM prepared_texts ORDER BY id")
    for pt_id, full_text, full_toks in cur.fetchall():
        if full_toks > CHUNK_THRESHOLD:
            for slice_txt in chunker(full_text):
                nt = len(tokenizer.tokenize(slice_txt))
                cur.execute(
                    """
                    INSERT INTO text_chunks
                      (text, tokenizer, nr_tokens, prepared_text_id)
                    VALUES (%s,%s,%s,%s)
                    """,
                    (slice_txt, TOKENIZER_NAME, nt, pt_id)
                )
        else:
            cur.execute(
                """
                INSERT INTO text_chunks
                  (text, tokenizer, nr_tokens, prepared_text_id)
                VALUES (%s,%s,%s,%s)
                """,
                (full_text, TOKENIZER_NAME, full_toks, pt_id)
            )

    conn.commit()
    cur.close(); conn.close()


# ─── STEP 3: METADATA XLSX → files + najdbe
def update_files_and_najdbe(xlsx_path: str):
    conn = _get_conn(); cur = conn.cursor()
    df = pd.read_excel(xlsx_path)

    for _, row in df.iterrows():
        s3key = row["file_name_with_folder"]
        year  = row["document_year"]
        vir   = row["Vir"].strip()

        cur.execute(
            "UPDATE files SET document_year = %s WHERE s3_key = %s RETURNING id",
            (year, s3key)
        )
        if cur.rowcount == 0:
            print(f"Warning: file not found for s3_key={s3key}")
            continue

        file_id = cur.fetchone()[0]
        cur.execute(
            "UPDATE najdbe SET file_id = %s WHERE vir = %s AND file_id IS NULL",
            (file_id, vir)
        )
        print(f"{cur.rowcount} najdbe rows linked for vir={vir}")

    conn.commit()
    cur.close(); conn.close()
