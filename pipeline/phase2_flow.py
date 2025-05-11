from pathlib import Path

from prefect import flow, task
from pipeline.phase2_preprocessing_and_inserting_json_data_into_db_tasks import (
    insert_file_and_sections,
    prepare_and_chunk_texts,
    update_files_and_najdbe,
)

@task(retries=3, retry_delay_seconds=2)
def run_insert(json_folder: str):
    folder = Path(json_folder)
    for jp in folder.glob("*.json"):
        insert_file_and_sections(str(jp))

@task(retries=3, retry_delay_seconds=2)
def run_prepare_and_chunk():
    prepare_and_chunk_texts()

@task(retries=3, retry_delay_seconds=2)
def run_update_metadata(xlsx_path: str):
    update_files_and_najdbe(xlsx_path)

@flow
def phase2_flow(
    json_input_folder: str = "output_jsons",
    metadata_xlsx_path: str = "./additional_metadata/test_vrst_in_virov.xlsx"
):
    run_insert(json_input_folder)
    run_prepare_and_chunk()
    run_update_metadata(metadata_xlsx_path)

if __name__ == "__main__":
    phase2_flow()
