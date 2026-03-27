from pathlib import Path
from prefect import flow, task
from pipeline.phase2_preprocessing_and_inserting_json_data_into_db_tasks import (
    insert_file_and_sections,
    prepare_and_chunk_texts,
    update_files_and_najdbe,
)

# Grabs all JSON files from the 'json_folder' directory
# and for each one calls insert_file_and_sections:
# - insert_file_and_sections reads JSON and inserts relevant data into tables files, sections, section_elements,
#   paragraphs, pictures, tables.
@task(retries=3, retry_delay_seconds=2)
def run_insert(json_folder: str):
    folder = Path(json_folder)
    for jp in folder.glob("*.json"):
        insert_file_and_sections(str(jp))

# Calls the prepare_and_chunk_texts function:
# - Merges short paragraphs from the same section and inserts them into prepared_texts.
# - Splits content of prepared_texts where possible into chunks and inserts them into text_chunks.
@task(retries=3, retry_delay_seconds=2)
def run_prepare_and_chunk():
    prepare_and_chunk_texts()

# Reads metadata from Excel (xlsx_path):
# - For each entry updates the year in the files table.
# - Then updates the najdbe table in rows that have the same source as the entry from files and do not yet have file_id.
@task(retries=3, retry_delay_seconds=2)
def run_update_metadata(xlsx_path: str):
    update_files_and_najdbe(xlsx_path)

# Prefect data flow that combines all three tasks from phase2:
# 1) run_insert: transfer and insertion of JSON data into the database.
# 2) run_prepare_and_chunk: preparation and splitting of texts.
# 3) run_update_metadata: updating additional metadata based on Excel file.
# Default paths to the folder with JSON files and path to the Excel file are defined in the function signature.
@flow
def phase2_flow(
    json_input_folder: str = "output_jsons",
    metadata_xlsx_path: str = "./additional_metadata/Seznam_vrst_in_virov_20241212.xlsx"
):
    run_insert(json_input_folder)
    run_prepare_and_chunk()
    run_update_metadata(metadata_xlsx_path)

if __name__ == "__main__":
    phase2_flow()