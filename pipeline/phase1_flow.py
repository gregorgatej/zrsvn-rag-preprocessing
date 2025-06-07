from prefect import flow, task
# Iz modula phase1 uvozimo dve funkciji: parse_pdf (prenese in pretvori PDF v JSON)
# ter extract_data (iz JSON izreže slike/tabele in pripravi končne JSONe).
from pipeline.phase1_generating_jsons_and_extracting_images_tasks import parse_pdf, extract_data
from pathlib import Path
import json

# Prefect naloga, ki:
# 1) Prebere seznam že obdelanih PDFov (podanih prek progress_file),
# 2) Pokliče parse_pdf, da prenese nove PDFje in jih pretvori v vmesne JSONe,
# 3) Shrani nove ključe (relativne poti do PDFjev) v progress_file,
# 4) Vrne slovar, kjer je ključ pot do PDFja, vrednost pa pot do ustvarjenega JSONa.
@task(retries=3, retry_delay_seconds=2)
def run_parse(pdf_input_folder: str, intermediate_folder: str, progress_file: str):
    # Load already processed files
    processed = load_progress(progress_file)
    result = parse_pdf(pdf_input_folder, intermediate_folder, already_processed=processed)
    save_progress(progress_file, list(result.keys()))
    return result

# Prefect naloga, ki:
# 1) Prebere seznam že obdelanih JSONov (podanih prek progress_file),
# 2) Pokliče extract_data, ki izravna slike/tabele in ustvari končne oblike JSON datotek,
# 3) Shrani nove ključe (relativne poti do PDF-jev) v progress_file,
# 4) Vrne slovar, kjer je ključ pot do PDFja, vrednost pa pot do končne oblike JSONa.
@task(retries=3, retry_delay_seconds=2)
def run_extract(parse_result: dict, base_dir: str, pics_out: str, json_out: str, progress_file: str):
    processed = load_progress(progress_file)
    result = extract_data(parse_result, base_dir=base_dir, pics_and_tables_out_path=pics_out, json_out_path=json_out, already_processed=processed)
    save_progress(progress_file, list(result.keys()))
    return result

# Prebere JSON datoteko s seznami že obdelanih ključev.
# - Če datoteka obstaja, vrne množico (set) že obdelanih ključev.
# - Če datoteka ne obstaja, vrne prazno množico.
def load_progress(progress_file):
    if Path(progress_file).exists():
        with open(progress_file, "r") as f:
            return set(json.load(f))
    return set()

# Shrani napredek:
# 1) Prebere obstoječo množico ključev (če obstaja).
# 2) Jo združi z novimi ključi (new_files).
# 3) Zapiše nazaj v JSON datoteko (kot sortiran seznam).
def save_progress(progress_file, new_files):
    processed = load_progress(progress_file)
    processed.update(new_files)
    with open(progress_file, "w") as f:
        json.dump(sorted(processed), f, indent=2)

# Prefect podatkovni tok, ki združi obe uvoženi nalogi iz phase1:
# 1) run_parse: prenese in pretvori PDFe v vmesne JSONe.
# 2) run_extract: iz vmesnih JSON-ov izreže slike/tabele in ustvari končne JSONe.
# Parametri so privzeti imeniki in datoteke za shranjevanje napredka.
@flow
def phase1_flow(pdf_input_folder: str = "input_pdfs",
                intermediate_folder: str = "intermediate_jsons",
                output_pics_folder: str = "output_pics_and_tables",
                output_json_folder: str = "output_jsons",
                base_dir: str = ".",
                parse_progress: str = "parse_progress.json",
                extract_progress: str = "extract_progress.json"):
    parse_result = run_parse(pdf_input_folder, intermediate_folder, parse_progress)
    processed_json_map = run_extract(parse_result, base_dir, output_pics_folder, output_json_folder, extract_progress)
    # Vrne slovar, kjer je ključ pot do PDFja, vrednost pa končni JSON.
    return processed_json_map

if __name__ == "__main__":
    result = phase1_flow()
    print("Final mapping of processed JSON files:", result)