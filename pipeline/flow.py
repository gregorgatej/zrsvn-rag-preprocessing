from prefect import flow, task
from pipeline.pdf_tasks import parse_pdf, extract_data

@task
def run_parse(pdf_input_folder: str, intermediate_folder: str):
    return parse_pdf(pdf_input_folder, intermediate_folder)

@task
def run_extract(parse_result: dict, base_dir: str, pics_out: str, json_out: str):
    return extract_data(parse_result, base_dir=base_dir, pics_and_tables_out_path=pics_out, json_out_path=json_out)

@flow
def pdf_processing_flow(pdf_input_folder: str = "input_pdfs",
                        intermediate_folder: str = "intermediate_jsons",
                        output_pics_folder: str = "output_pics_and_tables",
                        output_json_folder: str = "output_jsons",
                        base_dir: str = "."):
    parse_result = run_parse(pdf_input_folder, intermediate_folder)
    processed_json_map = run_extract(parse_result, base_dir, output_pics_folder, output_json_folder)
    return processed_json_map

if __name__ == "__main__":
    result = pdf_processing_flow()
    print("Final mapping of processed JSON files:", result)
