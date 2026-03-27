# Prefect: Tool for defining data flows and tasks.
# For sequential execution of all four PDF preprocessing phases.
from prefect import flow
# Import of individual phases wrapped in Prefect data flow functions.
# Phase 1: PDF parsing and JSON generation.
from pipeline.phase1_flow import phase1_flow
# Phase 2: Data preprocessing and insertion into database.
from pipeline.phase2_flow import phase2_flow
# Phase 3: Generation of additional metadata (language lists, descriptions, keywords and summaries).
from pipeline.phase3_flow import phase3_flow
# Phase 4: Generation of embeddings.
from pipeline.phase4_flow import phase4_flow

# Prefect data flow that connects all four phases.
@flow
def all_phases_flow(
    # Set where input PDFs are located for phase 1.
    pdf_input_folder: str = "input_pdfs",
    # Where to save intermediate JSONs from phase 1.
    intermediate_folder: str = "intermediate_jsons",
    # Where to save extracted images/tables from phase 1.
    output_pics_folder: str = "output_pics_and_tables",
    # Where to save final JSONs from phase 1.
    output_json_folder: str = "output_jsons",
    # Base path for phase 1.
    base_dir: str = ".",
    # File for tracking progress of phase 1.1 (parse_pdf)
    parse_progress: str = "parse_progress.json",
    # File for tracking progress of phase 1.2 (extract_data).
    extract_progress: str = "extract_progress.json",
    # Where the final JSONs from phase 2 will be located.
    json_input_folder: str = "output_jsons",
    # Path to the Excel file needed for phase 2.
    metadata_xlsx_path: str = "./additional_metadata/Seznam_vrst_in_virov_20241212.xlsx",
):
    phase1_flow(
        pdf_input_folder=pdf_input_folder,
        intermediate_folder=intermediate_folder,
        output_pics_folder=output_pics_folder,
        output_json_folder=output_json_folder,
        base_dir=base_dir,
        parse_progress=parse_progress,
        extract_progress=extract_progress,
    )

    phase2_flow(
        json_input_folder=json_input_folder,
        metadata_xlsx_path=metadata_xlsx_path
    )

    phase3_flow()

    phase4_flow()

if __name__ == "__main__":
    all_phases_flow()