#For running all four phases in order
from prefect import flow

# Phase 1: JSON generation & PDF parsing
from pipeline.phase1_flow import phase1_flow

# Phase 2: Preprocessing & DB insertion
from pipeline.phase2_flow import phase2_flow

# Phase 3: Metadata enrichment
from pipeline.phase3_flow import phase3_flow

# Phase 4: Embedding generation
from pipeline.phase4_flow import phase4_flow

@flow
def all_phases_flow(
    # you can override each phase’s params here if needed
    pdf_input_folder: str = "input_pdfs",
    intermediate_folder: str = "intermediate_jsons",
    output_pics_folder: str = "output_pics_and_tables",
    output_json_folder: str = "output_jsons",
    base_dir: str = ".",
    parse_progress: str = "parse_progress.json",
    extract_progress: str = "extract_progress.json",
    json_input_folder: str = "output_jsons",
    metadata_xlsx_path: str = "./additional_metadata/test_vrst_in_virov.xlsx",
):
    # Phase 1
    phase1_flow(
        pdf_input_folder=pdf_input_folder,
        intermediate_folder=intermediate_folder,
        output_pics_folder=output_pics_folder,
        output_json_folder=output_json_folder,
        base_dir=base_dir,
        parse_progress=parse_progress,
        extract_progress=extract_progress,
    )

    # Phase 2
    phase2_flow(
        json_input_folder=json_input_folder,
        metadata_xlsx_path=metadata_xlsx_path
    )

    # Phase 3
    phase3_flow()

    # Phase 4
    phase4_flow()

if __name__ == "__main__":
    all_phases_flow()
