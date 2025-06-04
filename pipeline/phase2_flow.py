from pathlib import Path

from prefect import flow, task
# Uvozimo relevantne funkcije iz phase2.
from pipeline.phase2_preprocessing_and_inserting_json_data_into_db_tasks import (
    # Funkcija, ki iz JSON datoteke vstavi podatke o datoteki in sekcijah v bazo.
    insert_file_and_sections,
    # Združi in razbije besedila iz baze na manjše kose.
    prepare_and_chunk_texts,
    # Funkcija, ki prek Excela posodobi tabeli files in najdbe.
    update_files_and_najdbe,
)

# Prefect naloga, ki pograbi vse JSON datoteke iz mape 'json_folder'
# in za vsako pokliče insert_file_and_sections:
# - insert_file_and_sections prebere JSON in vstavi relevantne podatke v tabele files, sections, section_elements, 
#   paragraphs, pictures, tables.
@task(retries=3, retry_delay_seconds=2)
def run_insert(json_folder: str):
    folder = Path(json_folder)
    for jp in folder.glob("*.json"):
        insert_file_and_sections(str(jp))

# Prefect naloga, ki kliče funkcijo prepare_and_chunk_texts:
# - Združi kratke odstavke iste sekcije in jih vstavi v prepared_texts.
# - Razbije vsebino prepared_texts po možnosti na kose in jih vstavi v text_chunks.
@task(retries=3, retry_delay_seconds=2)
def run_prepare_and_chunk():
    prepare_and_chunk_texts()

# Prefect naloga, ki iz Excela (xlsx_path) prebere metapodatke:
# - Za vsak vnos posodobi leto v tabeli files.
# - Nato posodobi tabelo najdbe v vrsticah, ki imajo isti vir kot vnos iz files in še nimajo file_id.
@task(retries=3, retry_delay_seconds=2)
def run_update_metadata(xlsx_path: str):
    update_files_and_najdbe(xlsx_path)

# Prefect podatkovni tok, ki združi vse tri naloge iz phase2:
# 1) run_insert: prenos in vstavljanje JSON podatkov v bazo.
# 2) run_prepare_and_chunk: priprava in razbijanje besedil.
# 3) run_update_metadata: posodabljanje dodatnih metapodatkov na podlagi Excel datoteke.
# Privzeta pot do mape z JSON datotekami in pot do Excel datoteke sta določeni v podpisu funkcije.
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
