from prefect import flow, task
from pipeline.phase3_adding_metadata_tasks import (
    upload_pics_and_tables_to_s3,
    process_text_chunk_languages,
    process_picture_descriptions,
    process_table_descriptions,
    process_text_chunk_metadata,
    process_picture_summaries,
    process_table_summaries,
    process_section_metadata,
    process_file_metadata,
)

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
