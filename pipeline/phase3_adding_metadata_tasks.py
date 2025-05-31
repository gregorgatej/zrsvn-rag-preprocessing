import os
from datetime import timedelta
import base64
import requests

import instructor
import instructor.patch
import openai
from openai import AzureOpenAI
from minio import Minio
from minio.error import S3Error
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from typing import Annotated, Optional, List
from pydantic import BaseModel, Field, AfterValidator
from transformers import AutoTokenizer

# ─── CONFIG ────────────────────────────────────────────────────────────────────
load_dotenv()

DB_PARAMS = {
    "dbname":   "zrsvn",
    "user":     "ggatej-pg",
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host":     "localhost",
    "port":     "5432",
    "options":  "-c search_path=rag_najdbe"
}

# Azure OpenAI client
endpoint         = os.getenv("ZRSVN_AZURE_OPENAI_ENDPOINT")
subscription_key = os.getenv("ZRSVN_AZURE_OPENAI_KEY")
api_version      = "2024-12-01-preview"
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)
instructor.patch(client=client)

# MinIO client
s3_client = Minio(
    endpoint=os.getenv("ZRSVN_S3_ENDPOINT", "moja.shramba.arnes.si"),
    access_key=os.getenv("S3_ACCESS_KEY"),
    secret_key=os.getenv("S3_SECRET_ACCESS_KEY"),
    secure=True
)
BUCKET_NAME = "zrsvn-rag-najdbe"

# tokenizer for text‐chunk metadata
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")


# ─── Pydantic schemas ───────────────────────────────────────────────────────────
class LanguageValidator:
    allowed_languages = {'SI', 'SH', 'EN', 'IT', 'DE', 'OTHER'}
    allowed_values = ", ".join(allowed_languages)

    @classmethod
    def validate_languages(cls, v):
        for lang in v:
            if lang not in cls.allowed_languages:
                raise ValueError(f"Invalid language code: {lang}. Allowed values are {cls.allowed_values}.")
        return v

def validate_keywords(v):
    if len(v) > 10:
        raise ValueError("A maximum of 10 keywords is allowed.")
    if len(v) != len(set(v)):  # Compares the length of the list to the length of the set (duplicates removed)
        raise ValueError("No duplicate keywords are allowed.")
    return v

class Languages(BaseModel):
    languages: Annotated[
        Optional[List[str]], 
        AfterValidator(LanguageValidator.validate_languages)
    ] = Field(
        ..., 
        description=f"List of languages the text is written in. Allowed values are {LanguageValidator.allowed_values}"
    )

class ImageMetadata(BaseModel):
    keywords: Annotated[
        List[str], 
        AfterValidator(validate_keywords)
    ] = Field(
        ..., 
        description=(
            "You are a chatbot whose task is to generate keywords for the content of an image. "
            "Based on the content of the image, you will select up to 10 keywords that best describe the key elements and themes of the image. "
            "The keywords should be specific, relevant to the content, and in Slovenian. "
            "The keywords should be useful for searching in the system, where the user can search for related content based on these words. Please ensure there are no repetitions in the keywords list."
        )
    )
    description: str = Field(
        ..., 
        description=(
            "You are a chatbot whose task is to describe images. "
            "Your job is to generate a detailed description for the given image, which includes details from the image that are important for search. "
            "This description should be supplemented with a general overview, but it must include specific information and data from the image that are relevant for potential searches. "
            "The description should be in Slovenian."
            )
    )

class TextSummary(BaseModel):
    summary: str = Field(
        ..., 
        description=(
            "You are a chatbot whose task is to generate a summary of the given text. "
            "Your job is to create a concise summary that captures the key points and themes of the text, while maintaining its original meaning. "
            "The summary should be clear and comprehensive, without including unnecessary details. The summary should be in Slovenian."
        )
    )

class TextChunkMetadata(BaseModel):
    keywords: Annotated[
        List[str], 
        AfterValidator(validate_keywords)
    ] = Field(
        ..., 
        description=(
            "You are a chatbot whose task is to generate keywords for the given text. "
            "Based on the content of the text, you will select up to 10 keywords that best describe the key elements and themes of the text. "
            "The keywords should be specific, relevant to the content, and in Slovenian. "
            "The keywords should be useful for searching in the system, where the user can search for related content based on these words. "
            "Please ensure there are no repetitions in the keywords list."
        )
    )
    summary: str = Field(
        ..., 
        description=(
            "You are a chatbot whose task is to generate a summary of the given text. "
            "Your job is to create a concise summary that captures the key points and themes of the text, while maintaining its original meaning. "
            "The summary should be clear and comprehensive, without including unnecessary details. "
            "The summary should be in Slovenian."
        )
    )

class SectionMetadata(BaseModel):
    keywords: Annotated[
        List[str],
        AfterValidator(validate_keywords)
    ] = Field(
        ...,
        description=(
            "You are a chatbot whose task is to generate keywords for the given text. "
            "Based on the content of the text, you will select up to 10 keywords that best describe the key elements and themes of the text. "
            "The keywords should be specific, relevant to the content, and in Slovenian. "
            "The keywords should be useful for searching in the system, where the user can search for related content based on these words. "
            "Please ensure there are no repetitions in the keywords list."
        )
    )
    summary: str = Field(
        ...,
        description=(
            "You are a chatbot whose task is to generate a high-level summary of the given content. "
            "The content you will be provided with consists of summaries that relate to text paragraphs, "
            "descriptions of images and descriptions of tables. Consider this (the type of element the "
            "content relates to) when generating the overarching summary. To help you with this, each of "
            "the content elements will have a corresponding tag before it (one of '[paragraph_summary]', "
            "'[picture_description_summary]', '[table_description_summary]'). "
            "Your job is to create a concise and comprehensive summary that captures the main ideas, themes, "
            "and key points of the text as a whole. Focus on providing an overarching summary that links the "
            "core concepts together, while maintaining the original meaning. The summary should be clear "
            "and to the point, avoiding excessive detail. The summary should be in Slovenian."
        )
    )

class HighLevelMetadata(BaseModel):
    keywords: Annotated[
        List[str], 
        AfterValidator(validate_keywords)
    ] = Field(
        ..., 
        description=(
            "You are a chatbot whose task is to generate keywords for the given text. "
            "Based on the content of the text, you will select up to 10 keywords that best describe the key elements and themes of the text. "
            "The keywords should be specific, relevant to the content, and in Slovenian. "
            "The keywords should be useful for searching in the system, where the user can search for related content based on these words. "
            "Please ensure there are no repetitions in the keywords list."
        )
    )
    summary: str = Field(
        ..., 
        description=(
            "You are a chatbot whose task is to generate a high-level summary of the given content. "
            "Your job is to create a concise and comprehensive summary that captures the main ideas, themes, and key points of the text as a whole. "
            "Focus on providing an overarching summary that links the core concepts together, while maintaining the original meaning. "
            "The summary should be clear and to the point, avoiding excessive detail. The summary should be in Slovenian."
        )
    )


# ─── TASK FUNCTIONS ─────────────────────────────────────────────────────────────

def upload_pics_and_tables_to_s3():
    local_folder = "./output_pics_and_tables"
    bucket_name = "zrsvn-rag-najdbe"

    # derive a dynamic prefix from the folder name
    folder_prefix = os.path.basename(os.path.normpath(local_folder))

    try:
        if not s3_client.bucket_exists(bucket_name):
            s3_client.make_bucket(bucket_name)
            print(f"✅ Bucket '{bucket_name}' created.")

        for root, dirs, files in os.walk(local_folder):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, local_folder)
                object_name = f"{folder_prefix}/{relative_path.replace(os.path.sep, '/')}"

                s3_client.fput_object(
                    bucket_name,
                    object_name,
                    full_path
                )
                print(f"⬆️  Uploaded: {object_name}")

    except S3Error as e:
        print("❌ Napaka:", e)

def process_text_chunk_languages(db_params=DB_PARAMS, client=client):
    conn = psycopg2.connect(**db_params)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, text
                    FROM rag_najdbe.text_chunks
                    WHERE id NOT IN (
                      SELECT text_chunk_id FROM rag_najdbe.languages
                    )
                """)
                chunks = cur.fetchall()
                total = len(chunks)
                print("Adding language metadata ...")
                for idx, (chunk_id, text) in enumerate(chunks, start=1):
                    print(f"Progress: [{idx}/{total}] - Processing text chunk ID: {chunk_id}")
                    chat_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        response_model=Languages,
                        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
                        max_tokens=96,
                        max_retries=2
                    )
                    for lang_code in chat_response.languages:
                        cur.execute("""
                            INSERT INTO rag_najdbe.languages (language, text_chunk_id)
                            VALUES (%s, %s)
                            ON CONFLICT (text_chunk_id, language) DO NOTHING
                        """, (lang_code, chunk_id))
        print(f"Processed {total} text_chunks and inserted languages where needed.")
    except Exception as e:
        print("Error processing languages:", e)
    finally:
        conn.close()


def process_picture_descriptions(db_params=DB_PARAMS, client=client,
                                 s3_client=s3_client, bucket_name=BUCKET_NAME):
    conn = psycopg2.connect(**db_params)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT p.id, se.s3_key
                    FROM rag_najdbe.pictures p
                    JOIN rag_najdbe.section_elements se
                      ON p.section_element_id = se.id
                    WHERE p.keywords IS NULL OR p.description IS NULL
                """)
                pending = cur.fetchall()
                total = len(pending)
                print("Adding picture descriptions and keywords ...")
                for idx, (pic_id, file_key) in enumerate(pending, start=1):
                    print(f"Progress: [{idx}/{total}] - Processing file: {file_key}")
                    url = s3_client.presigned_get_object(bucket_name, file_key, expires=timedelta(hours=1))
                    resp = requests.get(url)
                    if resp.status_code != 200:
                        raise Exception(f"Failed to download image {file_key}: {resp.status_code}")
                    b64 = base64.b64encode(resp.content).decode('utf-8')
                    data_url = f"data:image/png;base64,{b64}"
                    chat_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        response_model=ImageMetadata,
                        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}]}],
                        #max_tokens=640,
                        max_tokens=1280,
                        max_retries=2
                    )
                    kws = chat_response.keywords
                    desc = chat_response.description
                    cur.execute("""
                        UPDATE rag_najdbe.pictures
                        SET keywords = %s, description = %s
                        WHERE id = %s
                    """, (",".join(kws), desc, pic_id))
        print(f"Processed {total} pictures and updated metadata where needed.")
    except Exception as e:
        print("Error processing picture descriptions:", e)
    finally:
        conn.close()


def process_table_descriptions(db_params=DB_PARAMS, client=client,
                               s3_client=s3_client, bucket_name=BUCKET_NAME):
    conn = psycopg2.connect(**db_params)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT t.id, se.s3_key
                    FROM rag_najdbe.tables AS t
                    JOIN rag_najdbe.section_elements AS se
                      ON t.section_element_id = se.id
                    WHERE t.keywords IS NULL OR t.description IS NULL
                """)
                pending = cur.fetchall()
                total = len(pending)
                print("Adding table descriptions and keywords ...")
                for idx, (table_id, file_key) in enumerate(pending, start=1):
                    print(f"Progress: [{idx}/{total}] - Processing file: {file_key}")
                    url = s3_client.presigned_get_object(bucket_name, file_key, expires=timedelta(hours=1))
                    resp = requests.get(url)
                    if resp.status_code != 200:
                        raise Exception(f"Failed to download table image {file_key}: {resp.status_code}")
                    b64 = base64.b64encode(resp.content).decode('utf-8')
                    data_url = f"data:image/png;base64,{b64}"
                    chat_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        response_model=ImageMetadata,
                        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}]}],
                        # max_tokens=640,
                        max_tokens=1280,
                        max_retries=2
                    )
                    kws = chat_response.keywords
                    desc = chat_response.description
                    cur.execute("""
                        UPDATE rag_najdbe.tables
                        SET keywords = %s, description = %s
                        WHERE id = %s
                    """, (",".join(kws), desc, table_id))
        print(f"Processed {total} tables and updated metadata where needed.")
    except Exception as e:
        print("Error processing table descriptions:", e)
    finally:
        conn.close()


def process_text_chunk_metadata(db_params=DB_PARAMS, client=client):
    conn = psycopg2.connect(**db_params)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, text
                    FROM rag_najdbe.text_chunks
                    WHERE keywords IS NULL OR summary IS NULL
                """)
                rows = cur.fetchall()
                total = len(rows)
                print("Adding text chunk summaries and keywords ...")
                for idx, (chunk_id, text) in enumerate(rows, start=1):
                    print(f"Progress: [{idx}/{total}] - Processing text chunk ID: {chunk_id}")
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        response_model=TextChunkMetadata,
                        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
                        # max_tokens=256,
                        max_tokens=512,
                        max_retries=2
                    )
                    kws = resp.keywords
                    summ = resp.summary
                    cur.execute("""
                        UPDATE rag_najdbe.text_chunks
                        SET keywords = %s, summary = %s
                        WHERE id = %s
                    """, (",".join(kws), summ, chunk_id))
        print(f"Processed {total} text_chunks for metadata.")
    finally:
        conn.close()


def process_picture_summaries(db_params=DB_PARAMS, client=client):
    conn = psycopg2.connect(**db_params)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, description
                    FROM rag_najdbe.pictures
                    WHERE description IS NOT NULL
                      AND summary IS NULL
                """)
                rows = cur.fetchall()
                total = len(rows)
                print("Adding picture summaries ...")
                for idx, (pic_id, desc) in enumerate(rows, start=1):
                    print(f"Progress: [{idx}/{total}] - Processing picture ID: {pic_id}")
                    sum_resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        response_model=TextSummary,
                        messages=[{"role": "user", "content": [{"type": "text", "text": desc}]}],
                        # max_tokens=128,
                        max_tokens=256,
                        max_retries=2
                    )
                    cur.execute("""
                        UPDATE rag_najdbe.pictures
                        SET summary = %s
                        WHERE id = %s
                    """, (sum_resp.summary, pic_id))
        print(f"Processed {total} pictures for summaries.")
    finally:
        conn.close()


def process_table_summaries(db_params=DB_PARAMS, client=client):
    conn = psycopg2.connect(**db_params)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, description
                    FROM rag_najdbe.tables
                    WHERE description IS NOT NULL
                      AND summary IS NULL
                """)
                rows = cur.fetchall()
                total = len(rows)
                print("Adding table summaries ...")
                for idx, (tbl_id, desc) in enumerate(rows, start=1):
                    print(f"Progress: [{idx}/{total}] - Processing table ID: {tbl_id}")
                    sum_resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        response_model=TextSummary,
                        messages=[{"role": "user", "content": [{"type": "text", "text": desc}]}],
                        # max_tokens=128,
                        max_tokens=256,
                        max_retries=2
                    )
                    cur.execute("""
                        UPDATE rag_najdbe.tables
                        SET summary = %s
                        WHERE id = %s
                    """, (sum_resp.summary, tbl_id))
        print(f"Processed {len(rows)} tables for summaries.")
    finally:
        conn.close()


def process_section_metadata(db_params=DB_PARAMS, client=client):
    conn = psycopg2.connect(**db_params)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id
                    FROM rag_najdbe.sections
                    WHERE keywords IS NULL OR summary IS NULL
                    ORDER BY id
                """)
                section_ids = [r[0] for r in cur.fetchall()]
                total = len(section_ids)
                print("Adding section metadata ...")
                for idx, section_id in enumerate(section_ids, start=1):
                    print(f"Progress: [{idx}/{total}] - Processing section ID: {section_id}")
                    # Store unique (source_identifier, formatted_summary) tuples for the current section
                    unique_summary_sources = set() 
                    
                    ordered_summaries_for_llm = [] # This will store summaries in document order, with uniqueness based on source

                    # First, get all section_elements for the current section, ordered by their sequence in the section
                    cur.execute("""
                        SELECT id, type
                        FROM rag_najdbe.section_elements
                        WHERE section_id = %s
                        ORDER BY section_seq_position -- This is the key for order preservation!
                    """, (section_id,))
                    section_elements_in_order = cur.fetchall() # Returns list of (id, type)

                    for element_id, element_type in section_elements_in_order:
                        # This list will store tuples of (source_id, summary_text) for the current element
                        # The source_id will be text_chunk.id, picture.id, or table.id
                        current_element_summary_data = []

                        if element_type == 'paragraph':
                            cur.execute("""
                                SELECT tc.id, tc.summary
                                FROM rag_najdbe.paragraphs p
                                JOIN rag_najdbe.prepared_texts pt ON p.prepared_text_id = pt.id
                                JOIN rag_najdbe.text_chunks tc ON tc.prepared_text_id = pt.id
                                WHERE p.section_element_id = %s AND tc.summary IS NOT NULL
                                ORDER BY tc.id
                            """, (element_id,))
                            current_element_summary_data = cur.fetchall() # Returns list of (tc.id, tc.summary)

                        elif element_type == 'picture':
                            cur.execute("""
                                SELECT pic.id, pic.summary
                                FROM rag_najdbe.pictures pic
                                WHERE pic.section_element_id = %s AND pic.summary IS NOT NULL
                            """, (element_id,))
                            current_element_summary_data = cur.fetchall() # Returns list of (pic.id, pic.summary)

                        elif element_type == 'table':
                            cur.execute("""
                                SELECT tab.id, tab.summary
                                FROM rag_najdbe.tables tab
                                WHERE tab.section_element_id = %s AND tab.summary IS NOT NULL
                            """, (element_id,))
                            current_element_summary_data = cur.fetchall() # Returns list of (tab.id, tab.summary)
                        
                        tag_map = {
                            'paragraph': '[paragraph_summary]:\n',
                            'picture'  : '[picture_description_summary]:\n',
                            'table'    : '[table_description_summary]:\n',
                        }
                        prefix = tag_map[element_type]
                        
                        for source_id, summary_text in current_element_summary_data:
                            formatted_summary_for_llm = f"{prefix}{summary_text}"
                            
                            # Use a tuple (source_id, summary_text) to check for uniqueness
                            # This means "summary A from text_chunk 1" is different from "summary A from text_chunk 2"
                            uniqueness_key = (source_id, summary_text) 
                            
                            if uniqueness_key not in unique_summary_sources:
                                ordered_summaries_for_llm.append(formatted_summary_for_llm)
                                unique_summary_sources.add(uniqueness_key)

                    if not ordered_summaries_for_llm:
                        continue
                    
                    text = "\n\n".join(ordered_summaries_for_llm) # This 'text' will now preserve original order
                    print(f"\nText being passed into LLM:\n{text}\n")
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        response_model=SectionMetadata,
                        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
                        max_tokens=512,
                        max_retries=2
                    )
                    kws = resp.keywords
                    summ = resp.summary
                    cur.execute("""
                        UPDATE rag_najdbe.sections
                        SET keywords = %s, summary = %s
                        WHERE id = %s
                    """, (",".join(kws), summ, section_id))
        print(f"Processed {total} sections for high-level metadata.")
    finally:
        conn.close()


def process_file_metadata(db_params=DB_PARAMS, client=client):
    conn = psycopg2.connect(**db_params)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id
                    FROM rag_najdbe.files
                    WHERE keywords IS NULL OR summary IS NULL
                    ORDER BY id
                """)
                file_ids = [r[0] for r in cur.fetchall()]
                total = len(file_ids)
                print("Adding file metadata ...")
                for idx, file_id in enumerate(file_ids, start=1):
                    print(f"Progress: [{idx}/{total}] - Processing file ID: {file_id}")
                    # Modified SQL query to order sections by their first element's file_seq_position
                    cur.execute("""
                        SELECT s.summary
                        FROM rag_najdbe.sections s
                        JOIN rag_najdbe.section_elements se ON s.id = se.section_id
                        WHERE s.file_id = %s
                          AND s.summary IS NOT NULL
                        GROUP BY s.id, s.summary -- Group by s.id and s.summary to select s.summary
                        ORDER BY MIN(se.file_seq_position) -- Order sections by the earliest position of their elements
                    """, (file_id,))
                    parts = [r[0] for r in cur.fetchall()]
                    if not parts:
                        continue
                    text = " ".join(parts)
                    print(f"\nText being passed into LLM:\n{text}\n")
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        response_model=HighLevelMetadata,
                        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
                        # max_tokens=512,
                        max_tokens=1024,
                        max_retries=2
                    )
                    kws = resp.keywords
                    summ = resp.summary
                    cur.execute("""
                        UPDATE rag_najdbe.files
                        SET keywords = %s, summary = %s
                        WHERE id = %s
                    """, (",".join(kws), summ, file_id))
        print(f"Processed {total} files for high-level metadata.")
    finally:
        conn.close()