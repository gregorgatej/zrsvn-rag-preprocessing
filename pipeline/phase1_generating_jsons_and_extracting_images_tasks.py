# Delo z JSON.
import json
# Generiranje MD5 ali drugih zgoščenih vrednosti za unikatna imena.
import hashlib
# Beleženje informacij, opozoril in napak med izvajanjem.
import logging
# Merjenje časa, npr. za časovne žige in merjenje trajanja operacij.
import time
# Delo z datotečnimi potmi in preverjanje obstoja datotek.
from pathlib import Path
# Sistemske operacije, kot so ustvarjanje map.
import os
# Naključno generiranje, v našem primeru nizov znakov črk in števil. 
import random
# Dostop do niza črk in številk za ustvarjanje naključnih nizov.
import string
# Omogoča obravnavo "datotek", shranjenih v spominu"
import io
# Prikaz sprotnega napredka operacij.
from tqdm import tqdm
# PyMuPDF: odpiranje in obdelava PDFjev.
import fitz
# Pillow: odpiranje, obdelava in shranjevanje slik (PNG, JPEG ipd.).
from PIL import Image
# Docling: pretvorba dokumentov v interne modele.
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
# Klient za MinIO/S3 API.
from minio import Minio
# Ujemanje napak, ki jih vrže MinIO.
from minio.error import S3Error
# Nalaganje okoljskiih spremenljivk iz .env datoteke.
from dotenv import load_dotenv

# Nastavimo beležnik (ang. logger) za ta modul.
_log = logging.getLogger(__name__)
# V konzoli prikazujemo sporočila iz INFO ali višjega nivoja.
logging.basicConfig(level=logging.INFO)

# Funkcija, ki:
# 1) Prenese vse PDF-je iz MinIO/S3 (bucket 'zrsvn-rag-najdbe')
#    v lokalno mapo 'temp_input_pdfs'.
# 2) Za vsak prenesen PDF pokliče Docling DocumentConverter 
#    in ga pretvori v Docling model (tj. podatkovna struktura dostopna prek
#    'conv_result.document', ki predstalja vsebino dokumenta).
# 3) Model izvozi v obliki JSON datoteke v mapo 'intermediate_jsons'.
# 4) Vrne slovar, ki za vsako relativno pot do PDF-ja 
#    prikaže relativno pot do ustvarjenega JSON-a.
def parse_pdf(pdf_in_path, json_out_path="intermediate_jsons", already_processed: set = None):
    load_dotenv()
    s3_access_key = os.getenv("S3_ACCESS_KEY")
    s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
    s3_endpoint = "moja.shramba.arnes.si"
    # bucket_name = "zrsvn-rag-najdbe"
    bucket_name = "zrsvn-rag-najdbe-najvecji"
    client = Minio(
        endpoint=s3_endpoint,
        access_key=s3_access_key,
        secret_key=s3_secret_access_key,
        secure=True
    )

    # Ustvarimo (če še ne obstaja) mapo za prenesene PDFje.
    temp_input_folder = Path("temp_input_pdfs")
    temp_input_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Pridobimo seznam vseh objektov v S3 vedru (ang. bucketu).
        objects = client.list_objects(bucket_name, recursive=True)
        # Prenesemo vsak PDF, pri čemer sproti prek tqdm prikazujemo napredek.
        for obj in tqdm(objects, desc="Downloading PDFs from S3"):
            # Prenašamo samo PDFje.
            if obj.object_name.lower().endswith(".pdf"):
                local_file_path = temp_input_folder / obj.object_name
                # Če vedro vsebuje podmape jih ustvarimo tudi na disku.
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                client.fget_object(bucket_name, obj.object_name, str(local_file_path))
    except S3Error as e:
        _log.error(f"S3 error: {e}")

    pdf_in_path = temp_input_folder

     # Prepričamo se, da mapa za JSON izhod obstaja.
    json_out_path = Path(json_out_path)
    json_out_path.mkdir(parents=True, exist_ok=True)

    # Nastavimo parametre za Docling PDF pretvorbo.
    pipeline_options = PdfPipelineOptions()
    # Ne izvajamo optičnega prepoznavanja znakov.
    pipeline_options.do_ocr = False
    # Želimo prepoznati strukturo tabel.
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=32, device=AcceleratorDevice.CPU
    )
    # Inicializiramo DocumentConverter za PDF s PyPdfium zalednim delom.
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
    )

    # Ta slovar bomo napolnili s ključem=relativna pot do PDF, 
    # vrednost=relativna pot do JSON.
    result_dict = {}

    # Poiščemo vse PDF datoteke (rekurzivno) v temp_input_folder.
    # Zagotovimo, da je pdf_files vedno seznam poti do PDF-jev, bodisi:
    # - en element, če se pod pdf_in_path nahaja ena datoteka,
    # - seznam vseh PDF datotek, če se pod pdf_in_path nahaja mapa.
    pdf_files = (
        [pdf_in_path] if pdf_in_path.is_file()
        else list(pdf_in_path.rglob("*.pdf"))
    )

    for file in tqdm(pdf_files, desc="Parsing PDFs"):
        # Uporabimo relativno pot glede na trenutno delovno mapo
        pdf_dict_key = file.resolve().relative_to(Path.cwd()).as_posix()

        # Že obdelane datoteke preskočimo.
        if already_processed and pdf_dict_key in already_processed:
            _log.info(f"Skipping {pdf_dict_key}, already processed.")
            continue

        # Kreiramo unikatno ime JSON datoteke, ki vsebuje MD5 zgoščeno vrednost.
        hash_input = f"{file.resolve()}_{time.time()}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        max_length = 150
        suffix_part = f"_{hash_suffix}.json"
        base_stem = file.stem[: max_length - len(suffix_part)]
        output_filename = f"{base_stem}{suffix_part}"
        output_path = json_out_path / output_filename
        relative_json_path = f"{json_out_path.name}/{output_filename}"

        _log.info(f"Processing {pdf_dict_key}")
        start_time = time.time()
        conv_result = doc_converter.convert(file)
        elapsed_time = time.time() - start_time
        _log.info(f"Converted in {elapsed_time:.2f} seconds.")

        # Model, izvožen v obliki slovarja, shranimo kot JSON.
        with output_path.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps(conv_result.document.export_to_dict()))

        result_dict[pdf_dict_key] = relative_json_path

    return result_dict

# Funkcija, ki:
# 1) Za vsako JSON datoteko (rezultat parse_pdf) prebere podatke.
# 2) Iz podatkov 'pictures' in 'tables' izreže ustrezne slike s strani PDFja.
# 3) Shrani izrezane slike v mapo pics_and_tables_out_path.
# 4) Iz celotnega nabora besedil, slik in tabel oblikuje končne besedilne enote
#    oz. 'chunk' objekte, jih razporedi po straneh in doda metapodatke.
# 5) Shrani končne JSON datoteke v mapo json_out_path.
# 6) Vrne slovar, ki povezuje relativno pot do PDFja z relativno potjo do 
# končnega JSONa.
def extract_data(
    parse_pdf_result_dict,
    base_dir=".",
    pics_and_tables_out_path="output_pics_and_tables",
    json_out_path="output_json",
    already_processed: set = None
):
    # Poskrbimo, da izhodne mape obstajajo.
    os.makedirs(pics_and_tables_out_path, exist_ok=True)
    os.makedirs(json_out_path, exist_ok=True)

    # Za vsako PDF pot shranimo ustrezno zgoščeno vrednost (ang. hash) za imena.
    hash_suffix_map = {}
    # Omejitev dolžine imena, za v izogib omejitvam datotečnih sistemov.
    max_filename_length = 150

    # Pomožna funkcija za izrez slik:
    # - Za vsako referenco v pdf_dict nastavi (ključ = pot do PDF, 
    #                                          vrednost = pot do intermediate JSON).
    # - Odpre JSON, pridobi seznam slik in tabel (item.get("pictures"), 
    #                                             item.get("tables")).
    # - Za vsak element poišče robni okvir (ang. bounding box), 
    #   pri čemer upošteva tudi ostale z njim povezane, t.i. 'children' elemente,
    #   prek katerih razpotegne robni okvir.
    # - Izreže ustrezen del strani v slikovni obliki in ga hrani kot ločeno PNG 
    #   datoteko.
    # - V slovar item_bboxes shrani koordinate robnih okvirjev (ki zajemajo prostor,
    #   ki ga zaseda tako glavni element kot tisti, ki so z njim povezani).
    # - V slovar item_local_paths shrani lokalno pot do izrezane slike.
    def extract_pics_and_tables_helper(pdf_dict, base_dir, output_dir):
        # Ključ: (pdf_rel_path, tip, indeks, številka strani).
        # Vrednost: Koordinate robnega okvirja.
        item_bboxes = {}
        # Ključ: Isti kot zgoraj, tj. (pdf_rel_path, tip, indeks, številka strani).
        # Vrednost: Pot do slike, ki je bila shranjena na disk.
        item_local_paths = {}
        # Za vsak element (sliko ali tabelo) v 'items' (JSON strukturi):
        # - Poiščemo robni okvir.
        # - Če ima element "otroke" ($ref -> '#/texts/...'), razširimo robni okvir, 
        #   da zajamemo tudi morebitno pripadajoče besedilo.
        # - Preračunamo koordinate iz PDF točk v piksle (množitelj dpi_scale=2).
        # - Izrežemo del slike (crop) in ga shranimo kot PNG.
        # - Zabeležimo koordinate robnega okvirja in lokalno pot v slovarja.
        def process_items(items, item_type, doc, data, pdf_path, pdf_rel_path, hash_suffix):
            # Ime PDFja brez .pdf končnice.
            pdf_name = Path(pdf_path).stem
            for idx, item in enumerate(items):
                # 'prov' vsebuje informacije o lokaciji elementa (page_no, bbox).
                for prov in item.get("prov", []):
                    page_no = prov["page_no"]
                    page_index = page_no - 1
                    # Naložimo stran kot fitz Page objekt.
                    page = doc[page_index]
                    page_width = page.rect.width
                    page_height = page.rect.height
                    # Stran zrenderiramo v višji, dvakratni resoluciji.
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                     # pixmap pretvorimo v PIL sliko.
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    # Zberemo vse x in y koordinate robnega okvirja.
                    all_x = [prov["bbox"]["l"], prov["bbox"]["r"]]
                    all_y = [prov["bbox"]["t"], prov["bbox"]["b"]]
                    # Če ima element 'otroke', razširimo robni okvir, da zajame tudi te, tj. besedilo.
                    for child in item.get("children", []):
                        if "$ref" in child and child["$ref"].startswith("#/texts/"):
                            text_id = int(child["$ref"].split("/")[-1])
                            text_data = data["texts"][text_id]
                            for text_prov in text_data.get("prov", []):
                                if text_prov["page_no"] == page_no:
                                    all_x.extend([text_prov["bbox"]["l"], text_prov["bbox"]["r"]])
                                    all_y.extend([text_prov["bbox"]["t"], text_prov["bbox"]["b"]])
                    # Najdemo skrajne koordinate robnega okvirja (izražene v PDF točkah).
                    l, r = min(all_x), max(all_x)
                    bottom_pdf, top_pdf = min(all_y), max(all_y)
                    # Pretvorimo jih v piksle.
                    dpi_scale = 2
                    l_px = int(l * dpi_scale)
                    r_px = int(r * dpi_scale)
                    # Pri y koordinatah upoštevamo obratno koordinatno izhodišče.
                    t_px = int((page_height - top_pdf) * dpi_scale)
                    b_px = int((page_height - bottom_pdf) * dpi_scale)
                    crop_box = (l_px, t_px, r_px, b_px)
                    # Sestavimo ime izrezane slike glede na tip (picture/table) in indeks elementa.
                    if item_type == "picture":
                        suffix_part = f"_pic{idx+1}_pg{page_no}_{hash_suffix}.png"
                    else:
                        suffix_part = f"_tab{idx+1}_pg{page_no}_{hash_suffix}.png"
                    base_stem = pdf_name[: max_filename_length - len(suffix_part)]
                    output_filename = f"{base_stem}{suffix_part}"
                    output_path = os.path.join(output_dir, output_filename)
                    # Izrežemo (crop) sliko in jo shranimo v datoteko.
                    cropped_img = img.crop(crop_box)
                    cropped_img.save(output_path)
                    # Shranimo robni okvir (v PDF točkah) v slovar.
                    item_bboxes[(pdf_rel_path, item_type, idx, page_no)] = (l, top_pdf, r, bottom_pdf)
                    # Shranimo pot do lokalno shranjene slike v slovar.
                    item_local_paths[(pdf_rel_path, item_type, idx, page_no)] = output_path

        # Glavna zanka: gremo čez vsak PDF in pripadajoči JSON.
        for pdf_rel_path, json_rel_path in tqdm(pdf_dict.items(), desc="Extracting pics/tables"):
            pdf_path = str(Path(base_dir, pdf_rel_path))
            json_path = str(Path(base_dir, json_rel_path))
            # Preberemo 'vmesni' JSON (rezultat od parse_pdf).
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Odpremo PDF datoteko kot fitz dokument.
            doc = fitz.open(pdf_path)
            # Generiramo zgoščeno vrednost za ta PDF.
            hash_input = f"{Path(pdf_path).resolve()}_{time.time()}"
            hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            # Izrežemo slike iz "pictures" in tabele iz "tables".
            process_items(data.get("pictures", []), "picture", doc, data, pdf_path, pdf_rel_path, hash_suffix)
            process_items(data.get("tables", []), "table", doc, data, pdf_path, pdf_rel_path, hash_suffix)
            # Shranimo zgoščeno vrednost, da jo lahko kasneje uporabimo pri poimenovanju JSONov.
            hash_suffix_map[pdf_rel_path] = hash_suffix
            doc.close()
        return item_bboxes, item_local_paths

    # Ustvari naključen niz (ID) dolžine 9 znakov, sestavljen iz velikih in malih črk ter številk.
    # - random.choices izbere naključnih 'k' znakov iz niza (ascii_letters + digits).
    # - ''.join(...): združi izbrane znake v en niz.
    def generate_id():
        return ''.join(random.choices(string.ascii_letters + string.digits, k=9))

    # Ustvari slovar (chunk_dict) z vsemi relevantnimi podatki, ki se tičejo posameznega PDF elementa oz. 'chunk'-a:
    # - ref: referenca na izvorni JSON element (npr. "#/texts/3").
    # - boundingBox: slovar, ki vsebuje koordinate (l,t,r,b) in izvor koordinat.
    # - chunkID: enolični ID elementa (npr. "aB3kLm9Nz").
    # - contentType: seznam z enim elementom (npr. ["paragraph"], ["picture"], ["table"]).
    # - sectionPages: prazen seznam, ki se napolni kasneje.
    # - sectionID: ID sekcije, če element pripada kakšni sekciji.
    # - sectionHeader: besedilo naslova sekcije, če naslov obstaja.
    # - text: besedilo odstavka ali None, če gre za sliko/tabelo.
    # - nrCharacters: število znakov v odstavku (če contentType == "paragraph", sicer None).
    # - fileSeqPosition: pozicija elementa v celotnem dokumentu (se določi kasneje).
    # - sectionSeqPosition: pozicija elementa znotraj sekcije (se določi kasneje).
    # - chunkLocalPath: lokalna pot do izrezane slike (za slike/tabele) ali None.
    # Parametri:
    # - ref_value (str): referenca na izvorni JSON element (npr. "#/texts/3").
    # - content_type (str): en izmed 'paragraph', 'picture', 'table'.
    # - chunk_id (str): enolični ID, ustvarjen z generate_id().
    # - prov (dict): slovar z informacijami o elementu: {"page_no": int, "bbox": {...}, "charspan": [...]}.
    # - section_id (str ali None): ID sekcije, če obstaja.
    # - section_header (str ali None): besedilo naslova sekcije, če obstaja.
    # - text_value (str ali None): besedilo (če gre za odstavek).
    # - bounding_box_override (tuple ali None): koordinate robnega okvirja v PDF točkah (l, top, r, bottom).
    #   Ko obdelujemo slike in tabele (label == "picture" ali "table"), iz item_bboxes pridobimo
    #   razširjeni bounding box, če je ta podan. Če ni podan (tj. če je None) koordinate pridobimo privzete koordinate
    #   robnega okvirja prek prov["bbox"].
    # - chunk_local_path (str ali None): lokalna pot do izrezane slike (če je element tipa "picture" ali "table").
    # Vrne:
    # - chunk_dict (dict): slovar z vsemi zgoraj navedenimi ključi in vrednostmi.
    def create_chunk_entry(ref_value, content_type, chunk_id, prov, section_id, section_header, text_value, bounding_box_override=None, chunk_local_path=None):
        nr_chars = None
        # Če je chunk tipa odstavek, pridobimo charspan.
        if content_type == "paragraph":
            charspan = prov.get("charspan", [])
            if len(charspan) > 1:
                # Druga vrednost charspana nam da skupno število znakov.
                nr_chars = charspan[1]
        # Pripravimo bounding_box, bodisi iz bounding_box_override, bodisi iz prov["bbox"].
        if bounding_box_override is not None:
            # bounding_box_override je nabor vrednosti (l_pdf, top_pdf, r_pdf, bottom_pdf), izraženih v PDF točkah.
            (l_pdf, top_pdf, r_pdf, bottom_pdf) = bounding_box_override
            bounding_box = {
                "l": float(l_pdf),
                "t": float(top_pdf),
                "r": float(r_pdf),
                "b": float(bottom_pdf),
                "coord_origin": "BOTTOMLEFT"
            }
        else:
            # Uporabimo bbox vrednosti iz prov.
            bounding_box = {
                "l": prov["bbox"]["l"],
                "t": prov["bbox"]["t"],
                "r": prov["bbox"]["r"],
                "b": prov["bbox"]["b"],
                "coord_origin": "BOTTOMLEFT"
            }
        # Sestavimo slovar z vsemi vrednostmi.
        chunk_dict = {
            "ref": ref_value,
            "boundingBox": bounding_box,
            "chunkID": chunk_id,
            "contentType": [content_type],
            "sectionPages": [],
            "sectionID": section_id,
            "sectionHeader": section_header,
            "text": text_value,
            "nrCharacters": nr_chars,
            "fileSeqPosition": None,
            "sectionSeqPosition": None,
            "chunkLocalPath": chunk_local_path
        }
        return chunk_dict

    # Pridobimo slovarja item_bboxes in item_local_paths.
    item_bboxes, item_local_paths = extract_pics_and_tables_helper(
        pdf_dict=parse_pdf_result_dict,
        base_dir=base_dir,
        output_dir=pics_and_tables_out_path
    )
    
    #Pripravimo slovar, ki bo mapiral relativne poti vhodnih PDFov z relativnimi potmi izhodnih JSONov.
    pdf_processed_json_map = {}

    for pdf_rel_path, input_json_rel_path in tqdm(parse_pdf_result_dict.items(), desc="Processing final JSONs"):
        # Že obdelane PDFe preskočimo.
        if already_processed and pdf_rel_path in already_processed:
            _log.info(f"Skipping {pdf_rel_path}, already processed.")
            continue
        # TODO pdf_path lahko tu verjetno pobrišemo.
        pdf_path = str(Path(base_dir, pdf_rel_path))
        json_path = str(Path(base_dir, input_json_rel_path))
        # Preberemo vmesni JSON.
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Pridobimo ime datoteke in njen tip.
        file_name = data.get("origin", {}).get("filename", "unknown_file")
        file_type = file_name.split(".")[-1] if "." in file_name else "unknown"
        # Definiramo strukturo končnega JSON dokumenta.
        output_json = {
            "fileType": file_type,
            "fileName": file_name,
            "fileS3Path": str(Path(pdf_rel_path).relative_to("temp_input_pdfs")),
            # Tu kasneje dodamo vse podatke strani posamezne datoteke.
            "documentPages": []
        }
        # Zgradimo slovarje referenc za lažje iskanje znotraj posameznega sklopa.
        texts_dict = {}
        for i, txt in enumerate(data.get("texts", [])):
            ref_key = f"#/texts/{i}"
            texts_dict[ref_key] = txt
        pictures_dict = {}
        for i, pic in enumerate(data.get("pictures", [])):
            ref_key = f"#/pictures/{i}"
            pictures_dict[ref_key] = pic
        tables_dict = {}
        for i, tbl in enumerate(data.get("tables", [])):
            ref_key = f"#/tables/{i}"
            tables_dict[ref_key] = tbl
        groups_dict = {}
        for i, grp in enumerate(data.get("groups", [])):
            ref_key = f"#/groups/{i}"
            groups_dict[ref_key] = grp
        # Slovar za povezovanje IDja posamezne sekcije (sectionID) z množico strani, ki v njo spadajo.
        section_map = {}
        # Za hranjenje trenutno aktivne sekcije.
        current_section = None
        # Slovar, ki povezuje ID sekcije z naslovom sekcije.
        section_headers = {}
        # Seznam v katerega bomo shranili zbrane podatke posameznih elementov PDFa.
        final_chunks = []

        # Rekurzivna funkcija, ki:
        # - Pridobi referenco (npr. "#/texts/123") in ugotovi ali gre za besedilo, sliko, tabelo ali skupino.
        # - Če gre za 'section_header' ustvari novo sekcijo.
        # - Če gre za besedilo ('text' ali 'list_item') doda v final_chunks podatke o danem elementu PDFa (chunka);
        #   Če gre za sliko ali tabelo doda med podatke še koordinate robnega okvirja in lokalno pot do slike.
        # - Na koncu pokliče sebe za vse potomce (children) znotraj strukture JSON dokumenta.
        def traverse_node(ref_value):
            nonlocal current_section
            if ref_value in texts_dict:
                node = texts_dict[ref_value]
            elif ref_value in pictures_dict:
                node = pictures_dict[ref_value]
            elif ref_value in tables_dict:
                node = tables_dict[ref_value]
            elif ref_value in groups_dict:
                node = groups_dict[ref_value]
            else:
                return
            label = node.get("label", "")
            content_layer = node.get("content_layer", "")
            # Preskočimo neuporabno vsebino.
            if content_layer == "furniture":
                return
            if label == "section_header":
                new_section_id = generate_id()
                section_headers[new_section_id] = node.get("text", "")
                section_map[new_section_id] = set()
                current_section = new_section_id
            elif label in ["text", "list_item"]:
                if current_section is None:
                    new_section_id = generate_id()
                    section_headers[new_section_id] = None
                    section_map[new_section_id] = set()
                    current_section = new_section_id
                chunk_id = generate_id()
                section_id = current_section
                section_header = section_headers.get(section_id, None)
                for prov in node.get("prov", []):
                    page_no = prov["page_no"]
                    if section_id:
                        section_map[section_id].add(page_no)
                    # Za odstavke chunkLocalPath ostane null.
                    new_chunk = create_chunk_entry(
                        ref_value=ref_value,
                        content_type="paragraph",
                        chunk_id=chunk_id,
                        prov=prov,
                        section_id=section_id,
                        section_header=section_header,
                        text_value=node.get("text", "")
                    )
                    final_chunks.append((page_no, new_chunk))
            elif label == "picture":
                i = int(ref_value.split("/")[-1])
                chunk_id = generate_id()
                for prov in node.get("prov", []):
                    page_no = prov["page_no"]
                    bb_key = (pdf_rel_path, "picture", i, page_no)
                    bbox_override = item_bboxes.get(bb_key, None)
                    local_path = str(Path(item_local_paths.get(bb_key, None)).relative_to(base_dir))
                    new_chunk = create_chunk_entry(
                        ref_value=ref_value,
                        content_type="picture",
                        chunk_id=chunk_id,
                        prov=prov,
                        section_id=None,
                        section_header=None,
                        text_value=None,
                        bounding_box_override=bbox_override,
                        chunk_local_path=local_path
                    )
                    final_chunks.append((page_no, new_chunk))
            elif label == "table":
                i = int(ref_value.split("/")[-1])
                chunk_id = generate_id()
                for prov in node.get("prov", []):
                    page_no = prov["page_no"]
                    bb_key = (pdf_rel_path, "table", i, page_no)
                    bbox_override = item_bboxes.get(bb_key, None)
                    local_path = str(Path(item_local_paths.get(bb_key, None)).relative_to(base_dir))
                    new_chunk = create_chunk_entry(
                        ref_value=ref_value,
                        content_type="table",
                        chunk_id=chunk_id,
                        prov=prov,
                        section_id=None,
                        section_header=None,
                        text_value=None,
                        bounding_box_override=bbox_override,
                        chunk_local_path=local_path
                    )
                    final_chunks.append((page_no, new_chunk))
            # Rekurzivno obdelamo vse potomce (children).
            for child_info in node.get("children", []):
                child_ref = child_info.get("$ref", "")
                traverse_node(child_ref)

        # Prva faza: prehodimo vse otroke korenskega 'body' vozlišča.
        body = data.get("body", {})
        body_children = body.get("children", [])
        for child in body_children:
            child_ref = child.get("$ref", "")
            traverse_node(child_ref)

        # Druga faza: razporejanje slik in tabel v sekcije (če nimajo sekcije določene že od prej).
        last_text_section = None
        last_text_section_header = None
        no_section_pages = set()

        for i, (page_no, chunk) in enumerate(final_chunks):
            ctype = chunk["contentType"][0]
            if ctype == "paragraph":
                sec_id = chunk["sectionID"]
                if sec_id:
                    last_text_section = sec_id
                    last_text_section_header = chunk["sectionHeader"]
                else:
                    no_section_pages.add(page_no)
            elif ctype in ["picture", "table"]:
                if last_text_section is not None:
                    chunk["sectionID"] = last_text_section
                    chunk["sectionHeader"] = last_text_section_header
                    section_map[last_text_section].add(page_no)
                else:
                    no_section_pages.add(page_no)

        # Mapiramo sekcije s seznami strani.
        section_page_map = {}
        for sec_id, pages_set in section_map.items():
            section_page_map[sec_id] = sorted(pages_set)

        # Tretja faza: vsakemu chunku oz. elementu iz final_chunks napolnimo vsebino od 'sectionPages'.
        for i, (page_no, chunk) in enumerate(final_chunks):
            sec_id = chunk["sectionID"]
            if sec_id:
                chunk["sectionPages"] = section_page_map.get(sec_id, [])
            else:
                chunk["sectionPages"] = sorted(no_section_pages)

        # Določimo zaporedno pozicijo, ki jo chunk zaseda znotraj celotne PDF datoteke in zaporedno
        # pozicijo, ki jo zaseda znotraj sekcije.
        file_seq = 1
        section_counters = {}
        for page_no, chunk in final_chunks:
            chunk["fileSeqPosition"] = file_seq
            file_seq += 1
            sec_id = chunk["sectionID"]
            if sec_id not in section_counters:
                section_counters[sec_id] = 1
            else:
                section_counters[sec_id] += 1
            chunk["sectionSeqPosition"] = section_counters[sec_id]

        # Četrta faza: podatke združimo po straneh (pageNumber -> seznam chunkov).
        page_data = {}
        for page_no, chunk in final_chunks:
            if page_no not in page_data:
                page_data[page_no] = {
                    "pageNumber": page_no,
                    "chunks": []
                }
            page_data[page_no]["chunks"].append(chunk)

        # Strani sortiramo in jih vstavimo v končni slovar.
        sorted_page_numbers = sorted(page_data.keys())
        output_pages = [page_data[pn] for pn in sorted_page_numbers]
        output_json["documentPages"] = output_pages

        # Sestavimo ime končne JSON datoteke.
        json_file_name = os.path.basename(json_path)
        hash_suffix = hash_suffix_map.get(pdf_rel_path, "00000000")
        json_name = Path(json_file_name).stem
        suffix_part = f"_{hash_suffix}.json"
        base_stem = f"{json_name}"
        base_stem_trimmed = base_stem[: max_filename_length - len(suffix_part)]
        output_filename = f"{base_stem_trimmed}{suffix_part}"
        output_json_path = Path(json_out_path) / output_filename

        # Končni slovar shranimo v JSON datoteko.
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        _log.info(f"Processed JSON saved to {output_json_path}")
        pdf_processed_json_map[pdf_rel_path] = str(output_json_path)

    return pdf_processed_json_map