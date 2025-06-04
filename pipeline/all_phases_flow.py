# Za zaporedni zagon vseh štirih faz predprocesiranja PDF datotek.
from prefect import flow

# Uvoz posameznih faz, ki so ovite v Prefect funkcije podatkovnih tokov.
# Faza 1: Razčlenjevanje PDFjev in generiranje JSONov.
from pipeline.phase1_flow import phase1_flow
# Faza 2: Predobdelava podatkov in njihovo vstavljanje v bazo.
from pipeline.phase2_flow import phase2_flow
# Faza 3: Generiranje dodatnih metapodatkov (seznamov jezikov, opisov, ključnih besed in povzetkov).
from pipeline.phase3_flow import phase3_flow
# Faza 4: Generiranje vložitev.
from pipeline.phase4_flow import phase4_flow

# Prefect podatkovni tok, ki poveže vse štiri faze.
@flow
def all_phases_flow(
    # Privzeti parametri (ki jih lahko ob zagonu skripte tudi prepišemo).
    # Nastavimo kje se nahajajo vhodni PDFji za fazo 1.
    pdf_input_folder: str = "input_pdfs",
    # Kam shranimo vmesne JSONe iz faze 1.
    intermediate_folder: str = "intermediate_jsons",
    # Kam shranimo izrezane slike/tabele iz faze 1.
    output_pics_folder: str = "output_pics_and_tables",
    # Kam shranimo končne JSON-e iz faze 1.
    output_json_folder: str = "output_jsons",
    # Osnovna pot za fazo 1.
    base_dir: str = ".",
    # Datoteka za beleženje napredka faze 1.1 (parse_pdf)
    parse_progress: str = "parse_progress.json",
    # Datoteka za beleženje napredka faze 1.2 (extract_data).
    extract_progress: str = "extract_progress.json",
    # Kje se bodo nahajali končni JSONi faze 2.
    json_input_folder: str = "output_jsons",
    # Pot do Excel datoteke, ki jo potrebujemo za fazo 2.
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

# === Povzetek možnih optimizacij ===
# 1) Trenutni tok ne preverja uspeha vsake faze pred nadaljevanjem; dodati pogoje (if/else) ali Prefect wait_for, 
#    da zagotovimo pravilno zaporedje in preprečimo nepotrebne klice.