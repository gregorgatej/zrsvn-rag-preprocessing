import os
import logging
import numpy as np
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
from dotenv import load_dotenv
from tqdm import tqdm
# Za ustvarjanje vložitev (ang. embeddings).
from FlagEmbedding import FlagAutoModel

# Registriramo numpy adapterje, da lahko prek psycopg2 vstavimo v bazo podatke vezane na vložitve.
register_adapter(np.ndarray, lambda arr: AsIs(arr.tolist()))
register_adapter(np.float32, lambda val: AsIs(val))
register_adapter(np.float64, lambda val: AsIs(val))

load_dotenv()
DB_PARAMS = {
    "dbname":   "zrsvn",
    "user":     "ggatej-pg",
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host":     "localhost",
    "port":     "5432",
    "options":  "-c search_path=rag_najdbe"
}

# Definiramo parametre modela, ki ga uporabimo za generiranje vložitev.
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_MODEL_TYPE = "text"
EMBED_VECTOR_DIM = 768
BATCH_SIZE       = 100
# TODO Ta prompt bi lahko prilagodili.
QUERY_HINT       = "Represent this sentence for searching relevant passages:"

# Glavna funkcija, ki:
# 1) Shrani informacije o modelu v embedding_models (če tam še ni prisoten).
# 2) Naloži vložitveni model (prek FlagAutoModel).
# 3) Ustvari vložitve za:
#    a) text_chunks (besedilne bloke),
#    b) picture descriptions (opise slik),
#    c) table descriptions (opise tabel).
# 4) Shranjuje vložitve v tabelo embeddings.
def generate_and_store_embeddings():
    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(**DB_PARAMS)
    cur  = conn.cursor()

    # 1) Upsert embedding_models metadata
    cur.execute(
        "SELECT id FROM embedding_models WHERE name = %s",
        (EMBED_MODEL_NAME,)
    )
    row = cur.fetchone()
    if row:
        model_id = row[0]
        logging.info(f"Reusing embedding_model id={model_id}")
    else:
        cur.execute(
            """
            INSERT INTO embedding_models (type, name, vector_dimension)
            VALUES (%s, %s, %s) RETURNING id
            """,
            (EMBED_MODEL_TYPE, EMBED_MODEL_NAME, EMBED_VECTOR_DIM)
        )
        model_id = cur.fetchone()[0]
        conn.commit()
        logging.info(f"Inserted new embedding_model id={model_id}")

    # 2) Load the model once
    logging.info(f"Loading model {EMBED_MODEL_NAME}…")
    model = FlagAutoModel.from_finetuned(
        EMBED_MODEL_NAME,
        query_instruction_for_retrieval=QUERY_HINT,
        # Uporabimo standardno, "polno" natančnost številčnih izračunov.
        use_fp16=False,
        # Naprava, ki jo uporabimo je GPU na indeksu 0.
        device=["cuda:0"]
    )

    # TODO: Preveriti, ali je mogoče naložiti model na CPU, kadar GPU ni na voljo, da preprečimo napako.
    # TODO: Razmisliti o caching-u modela, če se funkcija kliče večkrat, namesto vsakič nalagati znova.

    # Obdelava besedilnih blokov.
    logging.info("Fetching text chunks…")
    cur.execute("SELECT id, text FROM text_chunks;")
    chunks = cur.fetchall()
    total = len(chunks)
    logging.info(f"{total} text chunks fetched")

    insert_tc = """
        INSERT INTO embeddings (vector, embedding_model_id, text_chunk_id)
        VALUES (%s, %s, %s)
    """
    # Seznam text_chunks obdelamo po skupinah (ang. batches), da ne preobremenimo pomnilnika.
    for i in tqdm(range(0, total, BATCH_SIZE), desc="TextChunk Batches"):
        batch = chunks[i : i + BATCH_SIZE]
        ids, texts = zip(*batch)
        out = model.encode(list(texts))
        vecs = np.array(out["dense_vecs"], dtype=np.float32)
        to_ins = [(v.tolist(), model_id, cid) for v, cid in zip(vecs, ids)]
        cur.executemany(insert_tc, to_ins)
        conn.commit()

    # Obdelava opisov slik.
    logging.info("Fetching picture descriptions…")
    cur.execute("SELECT id, description FROM pictures WHERE description IS NOT NULL;")
    pics = cur.fetchall()
    total = len(pics)
    logging.info(f"{total} picture descriptions fetched")

    insert_pic = """
        INSERT INTO embeddings (vector, embedding_model_id, picture_id)
        VALUES (%s, %s, %s)
    """
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Picture Batches"):
        batch = pics[i : i + BATCH_SIZE]
        ids, texts = zip(*batch)
        out = model.encode(list(texts))
        vecs = np.array(out["dense_vecs"], dtype=np.float32)
        to_ins = [(v.tolist(), model_id, pid) for v, pid in zip(vecs, ids)]
        cur.executemany(insert_pic, to_ins)
        conn.commit()

    # Obdelava opisov tabel.
    logging.info("Fetching table descriptions…")
    cur.execute("SELECT id, description FROM tables WHERE description IS NOT NULL;")
    tabs = cur.fetchall()
    total = len(tabs)
    logging.info(f"{total} table descriptions fetched")

    insert_tab = """
        INSERT INTO embeddings (vector, embedding_model_id, table_id)
        VALUES (%s, %s, %s)
    """
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Table Batches"):
        batch = tabs[i : i + BATCH_SIZE]
        ids, texts = zip(*batch)
        out = model.encode(list(texts))
        vecs = np.array(out["dense_vecs"], dtype=np.float32)
        to_ins = [(v.tolist(), model_id, tid) for v, tid in zip(vecs, ids)]
        cur.executemany(insert_tab, to_ins)
        conn.commit()

    cur.close()
    conn.close()
    logging.info("All embeddings generated and stored!")

if __name__ == "__main__":
    generate_and_store_embeddings()

# === Povzetek možnih optimizacij ===
# 1) UPSERT logika za vsako tabelo embeddings: trenutni INSERT lahko povzroči napako ali podvajanje,
#    če ponovno zaženemo skripto. Bolje bi bilo uporabiti "ON CONFLICT" ali predhodno preveriti, 
#    ali embedding že obstaja (npr. kombinacija model_id + text_chunk_id).
# 2) Batch size: trenutno je fiksno nastavljeno na 100. Odvisno od pomnilniških omejitev GPU-ja in baze,
#    bi lahko dinamično prilagajali velikost batch-a ali spremljali porabo pomnilnika.
# 3) Model naložimo le enkrat, kar je dobro. Lahko pa razmislimo o uporabi FP16 ali CPU mode, 
#    če GPU ni na voljo, da se izognemo napakam ali hitreje izvajamo na različnih strojnih okoljih.