from prefect import flow, task
from pipeline.phase3_adding_metadata_tasks import (
    # Funkcija za prenos izrezanih slik in tabel v S3.
    upload_pics_and_tables_to_s3,
    # Funkcija za ugotavljanje jezika posameznih text_chunk-ov
    process_text_chunk_languages,
    # Funkcija za generiranje opisov in ključnih besed za slike.
    process_picture_descriptions,
    # Funkcija za generiranje opisov in ključnih besed za tabele.
    process_table_descriptions,
    # Funkcija za generiranje povzetkov in ključnih besed za text_chunk-e.
    process_text_chunk_metadata,
    # Funkcija za generiranje povzetkov že opisanih slik.
    process_picture_summaries,
    # Funkcija za generiranje povzetkov že opisanih tabel.
    process_table_summaries,
    # Funkcija za generiranje povzetkov in ključnih besed za sekcije.
    process_section_metadata,
    # Funkcija za generiranje povzetkov in ključnih besed za celotne dokumente.
    process_file_metadata,
)

# Prefect naloge, ki kličejo vsako izmed uvoženih funkcij.
@task(retries=3, retry_delay_seconds=2)
def run_upload_pics_and_tables():
    upload_pics_and_tables_to_s3()

@task(retries=3, retry_delay_seconds=2)
def run_text_chunk_languages():
    process_text_chunk_languages()

@task(retries=3, retry_delay_seconds=2)
def run_picture_metadata():
    process_picture_descriptions()

@task(retries=3, retry_delay_seconds=2)
def run_table_metadata():
    process_table_descriptions()

@task(retries=3, retry_delay_seconds=2)
def run_text_chunk_metadata():
    process_text_chunk_metadata()

@task(retries=3, retry_delay_seconds=2)
def run_picture_summaries():
    process_picture_summaries()

@task(retries=3, retry_delay_seconds=2)
def run_table_summaries():
    process_table_summaries()

@task(retries=3, retry_delay_seconds=2)
def run_section_metadata():
    process_section_metadata()

@task(retries=3, retry_delay_seconds=2)
def run_file_metadata():
    process_file_metadata()

# Prefect podatkovni tok, ki poveže vse naloge tretje faze predprocesiranja dokumentov v spodnje zaporedje:
# 1) run_upload_pics_and_tables
# 2) run_text_chunk_languages
# 3) run_picture_metadata
# 4) run_table_metadata
# 5) run_text_chunk_metadata
# 6) run_picture_summaries
# 7) run_table_summaries
# 8) run_section_metadata
# 9) run_file_metadata
# Vsaka naloga se bo ob neuspeli poskusu ponovila do 3-krat (z dvosekundnim zamikom ned vsako ponovitvijo).
@flow
def phase3_flow():
    run_upload_pics_and_tables()
    run_text_chunk_languages()
    run_picture_metadata()
    run_table_metadata()
    run_text_chunk_metadata()
    run_picture_summaries()
    run_table_summaries()
    run_section_metadata()
    run_file_metadata()

if __name__ == "__main__":
    phase3_flow()