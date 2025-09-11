import os
import json
from pathlib import Path
import pandas as pd
# Povezava in delo s PostgreSQL bazo.
import psycopg2
# Prilagajanje tipov PostgreSQLu.
from psycopg2.extensions import register_adapter, AsIs
from dotenv import load_dotenv
from transformers import AutoTokenizer
import semchunk
import numpy as np

load_dotenv()
TOKENIZER_NAME = "BAAI/bge-m3"
SEMTEXT_CHUNK_SIZE = 512
# Če je število tokenov manjše od polovice SEMTEXT_CHUNK_SIZE, bomo združevali odstavke.
PREP_THRESHOLD   = SEMTEXT_CHUNK_SIZE / 2
# Če bo število tokenov več kot 1.5-kratnik SEMTEXT_CHUNK_SIZE, bomo razbijali besedilo na manjše dele.
CHUNK_THRESHOLD  = SEMTEXT_CHUNK_SIZE * 1.5

# Parametri za povezavo z PostgreSQL bazo.
DB_PARAMS = {
    "dbname":   "zrsvn",
    "user":     "ggatej-pg",
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host":     "localhost",
    "port":     "5432",
    "options":  "-c search_path=rag_najdbe"
}

# Registriramo adapterje, da PostgreSQL sprejme numpy tipe.
register_adapter(np.ndarray, lambda arr: AsIs(arr.tolist()))
register_adapter(np.float32, lambda val: AsIs(val))
register_adapter(np.float64, lambda val: AsIs(val))

# Ustvari in vrne novo povezavo do PostgreSQL baze s parametri DB_PARAMS.
def _get_conn():
    return psycopg2.connect(**DB_PARAMS)

# Prebere JSON datoteko (ki vsebuje informacije o dokumentu, sekcijah in chunkih) in nato:
# 1) Vstavi zapis v tabelo files (tip, ime, s3_key).
# 2) Za vsak chunk v JSONu:
#    a) Preveri, ali je sekcija s pripadajočim 'sectionID' že v bazi. Če ni, jo ustvari
#       v tabeli 'sections', kjer se hrani naslov (header) sekcije in povezava na datoteko (file_id).
#    b) V tabelo `section_elements` vstavi zapis, ki združuje podatke o chunku:
#       - tip (en izmed 'paragraph', 'picture', 'table'),
#       - zaporedni položaj v celotnem dokumentu in znotraj sekcije,
#       - številka strani,
#       - koordinate robnega okvirja, shranjene kot JSON niz,
#       - pot do lokalno shranjene slike (če je tip chunka slika ali tabela), ki je predstavlja njen S3 ključ, 
#       - povezava na ustrezno sekcijo (section_id).
#    c) Če je chunk odstavek (paragraph) vstavi v tabelo paragraphs njegovo besedilo, št. tokenov (nr_tokens),
#       št. znakov (nr_characters) in povezavo na to kateri element znotraj sekcije predstavlja (section_element_id).
#    d) Če je chunk slika, vstavi zgolj povezavo na to kateri element znotraj sekcije predstavlja (section_element_id).
#       Zapis vstavi v tabelo pictures.
#    e) Če je chunk tabela, vstavi zgolj povezavo na to kateri element znotraj sekcije predstavlja (section_element_id).
#       Zapis vstavi v tabelo tables.
def insert_file_and_sections(json_path: str):
    conn = _get_conn(); cur = conn.cursor()
    # Naložimo tokenizator (uporabljali ga bomo za štetje tokenov).
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    # Preberemo celotno vsebino JSON datoteke.
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    cur.execute(
        "INSERT INTO files (type, filename, s3_key) VALUES (%s,%s,%s) ON CONFLICT (s3_key) DO NOTHING RETURNING id",
        (data["fileType"], data["fileName"], data["fileS3Path"])
    )
    file_id = cur.fetchone()[0]

    # Slovar za mapiranje sectionIDja iz JSONa na ID sekcije v bazi.
    section_map = {}
    # Sprehodimo se po vseh straneh v JSONu.
    for page in data["documentPages"]:
        page_nr = page["pageNumber"]
        for chunk in page["chunks"]:
            sid_json = chunk["sectionID"]
            # Če na trenutno sekcijo še nismo naleteli jo vstavimo v tabelo sections.
            if sid_json not in section_map:
                cur.execute(
                    "INSERT INTO sections (header, file_id) VALUES (%s,%s) RETURNING id",
                    (chunk["sectionHeader"], file_id)
                )
                section_map[sid_json] = cur.fetchone()[0]

            se_id = section_map[sid_json]
            bbox = chunk["boundingBox"]
            # Če gre za sliko/tabelo je chunkLocalPath enak S3 ključu.
            s3key = chunk.get("chunkLocalPath")

            # Vstavimo zapis v section_elements.
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

# 1) Pridobimo vse odstavke iz tabele paragraphs in tiste, ki so manjši (in znotraj iste sekcije) združimo.
# 2) Originalne odstavke (v primeru, da so dovolj veliki) ali združeno besedilo vstavimo v tabelo prepared_texts.
# 3) Če je vnos iz prepared_text daljši od CHUNK_THRESHOLD ga razbijemo na manjše kose, sicer ga shranimo
#    v tabelo text_chunks takega kot je.
def prepare_and_chunk_texts():
    conn = _get_conn(); cur = conn.cursor()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    chunker = semchunk.chunkerify(tokenizer, SEMTEXT_CHUNK_SIZE)

    # Vse odstavke pridobimo v pravilnem vrstnem redu.
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

        # Če je odstavek prekratek ga združimo z naslednjim iz iste sekcije (če le-ta na voljo), 
        # dokler ne presežemo zastavljene meje (slednja je hevristično določena kot SEMTEXT_CHUNK_SIZE / 2,
        # pri čemer v naši konkretni implementaciji SEMTEXT_CHUNK_SIZE znaša 512.
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

        # Preparirano besedilo vstavimo v prepared_texts.
        cur.execute(
            """
            INSERT INTO prepared_texts
              (text, nr_tokens, text_chunk_size_threshold)
            VALUES (%s,%s,%s) RETURNING id
            """,
            (group_text, group_tokens, PREP_THRESHOLD)
        )
        pt_id = cur.fetchone()[0]

        # Vsak vnos prepariranega besedila zvežemo nazaj z odstavkom/-i na katerega/-e se navezuje.
        cur.execute(
            "UPDATE paragraphs SET prepared_text_id = %s WHERE id = ANY(%s)",
            (pt_id, group_ids)
        )
        i = next_i

    # Besedilo vsake enote iz prepared_text vstavimo v tabelo text_chunks, skupaj s podatki o tokenizatorju, številom
    # tokenov in IDjem. Če je besedilo daljše od hevristično določene meje (SEMTEXT_CHUNK_SIZE * 1.5) ga dodatno razbijemo
    # in vsakega izmed manjših delov besedila shranimo posebej.
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

# Prebere Excel datoteko z metapodatki (vrsticami, ki vsebujejo s3_key, leto in vir) in nato:
# 1) Za vsako vrstico iz Excel datoteke kjer se file_name_with_folder ujema z s3_key iz files vnese v slednjo
#    tabelo podatek o letnici dokumenta.
# 2) Kjer vira sovpadata in file_id še ni nastavljen posodobi povezave med vrsticami tabele najdbe in tabele files.
#    Na ta način vemo katere datoteke se navezujejo na katere najdbe vrst.
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