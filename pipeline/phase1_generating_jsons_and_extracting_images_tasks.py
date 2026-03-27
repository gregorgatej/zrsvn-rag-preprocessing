import json
import hashlib
import logging
import time
from pathlib import Path
import os
import random
import string
import io
from tqdm import tqdm
import fitz
from PIL import Image
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 1) Downloads all PDFs from MinIO/S3 (bucket 'zrsvn-rag-najdbe')
#    into local folder 'temp_input_pdfs'.
# 2) For each downloaded PDF, calls Docling DocumentConverter
#    and converts it to Docling model (i.e., a data structure accessible via
#    'conv_result.document', which represents the document contents).
# 3) Exports the model as JSON file into 'intermediate_jsons' folder.
# 4) Returns a dictionary that for each relative PDF path
#    shows the relative path to the created JSON.
def parse_pdf(pdf_in_path, json_out_path="intermediate_jsons", already_processed: set = None):
    load_dotenv()
    s3_access_key = os.getenv("S3_ACCESS_KEY")
    s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
    s3_endpoint = "moja.shramba.arnes.si"
    bucket_name = "zrsvn-rag-najdbe-najvecji"
    client = Minio(
        endpoint=s3_endpoint,
        access_key=s3_access_key,
        secret_key=s3_secret_access_key,
        secure=True
    )

    # Create (if it doesn't exist) folder for downloaded PDFs.
    temp_input_folder = Path("temp_input_pdfs")
    temp_input_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Get list of all objects in S3 bucket.
        objects = client.list_objects(bucket_name, recursive=True)
        # Download each PDF, showing progress via tqdm.
        for obj in tqdm(objects, desc="Downloading PDFs from S3"):
            # Download only PDFs.
            if obj.object_name.lower().endswith(".pdf"):
                local_file_path = temp_input_folder / obj.object_name
                # If bucket contains subfolders, create them on disk too.
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                client.fget_object(bucket_name, obj.object_name, str(local_file_path))
    except S3Error as e:
        _log.error(f"S3 error: {e}")

    pdf_in_path = temp_input_folder

    json_out_path = Path(json_out_path)
    json_out_path.mkdir(parents=True, exist_ok=True)

    # Set parameters for Docling PDF conversion.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    # We want to recognize table structure.
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=32, device=AcceleratorDevice.CPU
    )
    # Initialize DocumentConverter for PDF with PyPdfium backend.
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
    )

    # This dictionary will be filled with key=relative PDF path,
    # value=relative JSON path.
    result_dict = {}

    # Find all PDF files (recursively) in temp_input_folder.
    # Ensure pdf_files is always a list of PDF paths, either:
    # - one element if a single file is under pdf_in_path,
    # - list of all PDF files if a folder is under pdf_in_path.
    pdf_files = (
        [pdf_in_path] if pdf_in_path.is_file()
        else list(pdf_in_path.rglob("*.pdf"))
    )

    for file in tqdm(pdf_files, desc="Parsing PDFs"):
        pdf_dict_key = file.resolve().relative_to(Path.cwd()).as_posix()

        if already_processed and pdf_dict_key in already_processed:
            _log.info(f"Skipping {pdf_dict_key}, already processed.")
            continue

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

        with output_path.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps(conv_result.document.export_to_dict()))

        result_dict[pdf_dict_key] = relative_json_path

    return result_dict

# 1) For each JSON file (result of parse_pdf), reads the data.
# 2) From the 'pictures' and 'tables' data, extracts appropriate images from PDF pages.
# 3) Saves extracted images to pics_and_tables_out_path folder.
# 4) From the entire set of texts, images and tables forms final text units
#    or 'chunk' objects, arranges them by pages and adds metadata.
# 5) Saves final JSON files to json_out_path folder.
# 6) Returns a dictionary connecting relative PDF path to relative path to
# final JSON.
def extract_data(
    parse_pdf_result_dict,
    base_dir=".",
    pics_and_tables_out_path="output_pics_and_tables",
    json_out_path="output_json",
    already_processed: set = None
):
    os.makedirs(pics_and_tables_out_path, exist_ok=True)
    os.makedirs(json_out_path, exist_ok=True)

    hash_suffix_map = {}
    # Limit name length to avoid file system limitations.
    max_filename_length = 150

    # Helper function for extracting images:
    # - For each reference in pdf_dict, set (key = path to PDF,
    #                                          value = path to intermediate JSON).
    # - Open JSON, get list of images and tables (item.get("pictures"),
    #                                             item.get("tables")).
    # - For each element, find bounding box (ang. bounding box),
    #   considering other related 'children' elements,
    #   through which it extends the bounding box.
    # - Extract appropriate part of page as image and store as separate PNG
    #   file.
    # - In item_bboxes dictionary save bounding box coordinates (which cover the space
    #   occupied by both the main element and those related to it).
    # - In item_local_paths dictionary, save local path to extracted image.
    def extract_pics_and_tables_helper(pdf_dict, base_dir, output_dir):
        # Key: (pdf_rel_path, type, index, page number).
        # Value: Bounding box coordinates.
        item_bboxes = {}
        # Key: Same as above, i.e. (pdf_rel_path, type, index, page number).
        # Value: Path to image saved to disk.
        item_local_paths = {}
        # For each element (image or table) in 'items' (JSON structure):
        # - Find the bounding box.
        # - If element has "children" ($ref -> '#/texts/...'), extend bounding box,
        #   to capture also any associated text.
        # - Re-calculate coordinates from PDF points to pixels (multiplier dpi_scale=2).
        # - Extract part of image (crop) and save as PNG.
        # - Record bounding box coordinates and local path in dictionaries.
        def process_items(items, item_type, doc, data, pdf_path, pdf_rel_path, hash_suffix):
            # PDF name without .pdf extension.
            pdf_name = Path(pdf_path).stem
            for idx, item in enumerate(items):
                # 'prov' contains information about element location (page_no, bbox).
                for prov in item.get("prov", []):
                    page_no = prov["page_no"]
                    page_index = page_no - 1
                    # Load page as fitz Page object.
                    page = doc[page_index]
                    page_width = page.rect.width
                    page_height = page.rect.height
                    # Render page in higher, double resolution.
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                     # Convert pixmap to PIL image.
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    # Collect all x and y coordinates of bounding box.
                    all_x = [prov["bbox"]["l"], prov["bbox"]["r"]]
                    all_y = [prov["bbox"]["t"], prov["bbox"]["b"]]
                    # If element has 'children', extend bounding box to capture them too, i.e. text.
                    for child in item.get("children", []):
                        if "$ref" in child and child["$ref"].startswith("#/texts/"):
                            text_id = int(child["$ref"].split("/")[-1])
                            text_data = data["texts"][text_id]
                            for text_prov in text_data.get("prov", []):
                                if text_prov["page_no"] == page_no:
                                    all_x.extend([text_prov["bbox"]["l"], text_prov["bbox"]["r"]])
                                    all_y.extend([text_prov["bbox"]["t"], text_prov["bbox"]["b"]])
                    # Find extreme bounding box coordinates (expressed in PDF points).
                    l, r = min(all_x), max(all_x)
                    bottom_pdf, top_pdf = min(all_y), max(all_y)
                    # Convert them to pixels.
                    dpi_scale = 2
                    l_px = int(l * dpi_scale)
                    r_px = int(r * dpi_scale)
                    # For y coordinates, consider reverse coordinate origin.
                    t_px = int((page_height - top_pdf) * dpi_scale)
                    b_px = int((page_height - bottom_pdf) * dpi_scale)
                    crop_box = (l_px, t_px, r_px, b_px)
                    # Construct name of extracted image based on type (picture/table) and element index.
                    if item_type == "picture":
                        suffix_part = f"_pic{idx+1}_pg{page_no}_{hash_suffix}.png"
                    else:
                        suffix_part = f"_tab{idx+1}_pg{page_no}_{hash_suffix}.png"
                    base_stem = pdf_name[: max_filename_length - len(suffix_part)]
                    output_filename = f"{base_stem}{suffix_part}"
                    output_path = os.path.join(output_dir, output_filename)
                    # Extract (crop) image and save to file.
                    cropped_img = img.crop(crop_box)
                    cropped_img.save(output_path)
                    # Save bounding box (in PDF points) to dictionary.
                    item_bboxes[(pdf_rel_path, item_type, idx, page_no)] = (l, top_pdf, r, bottom_pdf)
                    # Save path to locally saved image in dictionary.
                    item_local_paths[(pdf_rel_path, item_type, idx, page_no)] = output_path

        # Main loop: iterate over each PDF and its associated JSON.
        for pdf_rel_path, json_rel_path in tqdm(pdf_dict.items(), desc="Extracting pics/tables"):
            pdf_path = str(Path(base_dir, pdf_rel_path))
            json_path = str(Path(base_dir, json_rel_path))
            # Read 'intermediate' JSON (result from parse_pdf).
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Open PDF file as fitz document.
            doc = fitz.open(pdf_path)
            # Generate hash value for this PDF.
            hash_input = f"{Path(pdf_path).resolve()}_{time.time()}"
            hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            # Extract images from "pictures" and tables from "tables".
            process_items(data.get("pictures", []), "picture", doc, data, pdf_path, pdf_rel_path, hash_suffix)
            process_items(data.get("tables", []), "table", doc, data, pdf_path, pdf_rel_path, hash_suffix)
            # Save hash value to use later for naming JSONs.
            hash_suffix_map[pdf_rel_path] = hash_suffix
            doc.close()
        return item_bboxes, item_local_paths

    # Generate a random string (ID) of 9 characters, composed of uppercase and lowercase letters and digits.
    # - random.choices select 'k' random characters from string (ascii_letters + digits).
    # - ''.join(...): combine selected characters into one string.
    def generate_id():
        return ''.join(random.choices(string.ascii_letters + string.digits, k=9))

    # Create dictionary (chunk_dict) with all relevant data concerning individual PDF element or 'chunk':
    # - ref: reference to source JSON element (e.g. "#/texts/3").
    # - boundingBox: dictionary containing coordinates (l,t,r,b) and coordinate origin.
    # - chunkID: unique element ID (e.g. "aB3kLm9Nz").
    # - contentType: list with one element (e.g. ["paragraph"], ["picture"], ["table"]).
    # - sectionPages: empty list, filled later.
    # - sectionID: section ID if element belongs to a section.
    # - sectionHeader: section title text if title exists.
    # - text: paragraph text or None if image/table.
    # - nrCharacters: number of characters in paragraph (if contentType == "paragraph", otherwise None).
    # - fileSeqPosition: element position in entire document (determined later).
    # - sectionSeqPosition: element position within section (determined later).
    # - chunkLocalPath: local path to extracted image (for images/tables) or None.
    # Parameters:
    # - ref_value (str): reference to source JSON element (e.g. "#/texts/3").
    # - content_type (str): one of 'paragraph', 'picture', 'table'.
    # - chunk_id (str): unique ID created with generate_id().
    # - prov (dict): dictionary with element information: {"page_no": int, "bbox": {...}, "charspan": [...]}.
    # - section_id (str or None): section ID if it exists.
    # - section_header (str or None): section title text if it exists.
    # - text_value (str or None): text (if it's a paragraph).
    # - bounding_box_override (tuple or None): bounding box coordinates in PDF points (l, top, r, bottom).
    #   When processing images and tables (label == "picture" or "table"), we obtain extended
    #   bounding box from item_bboxes if provided. If not provided (i.e. if it's None), we obtain default coordinates
    #   bounding box via prov["bbox"].
    # - chunk_local_path (str or None): local path to extracted image (if element type is "picture" or "table").
    # Returns:
    # - chunk_dict (dict): dictionary with all above keys and values.
    def create_chunk_entry(ref_value, content_type, chunk_id, prov, section_id, section_header, text_value, bounding_box_override=None, chunk_local_path=None):
        nr_chars = None
        # If chunk is paragraph type, get charspan.
        if content_type == "paragraph": 
            charspan = prov.get("charspan", [])
            if len(charspan) > 1:
                # Second value of charspan gives us total number of characters.
                nr_chars = charspan[1]
        # Prepare bounding_box, either from bounding_box_override or from prov["bbox"].
        if bounding_box_override is not None:
            # bounding_box_override is tuple (l_pdf, top_pdf, r_pdf, bottom_pdf), expressed in PDF points.
            (l_pdf, top_pdf, r_pdf, bottom_pdf) = bounding_box_override
            bounding_box = {
                "l": float(l_pdf),
                "t": float(top_pdf),
                "r": float(r_pdf),
                "b": float(bottom_pdf),
                "coord_origin": "BOTTOMLEFT"
            }
        else:
            bounding_box = {
                "l": prov["bbox"]["l"],
                "t": prov["bbox"]["t"],
                "r": prov["bbox"]["r"],
                "b": prov["bbox"]["b"],
                "coord_origin": "BOTTOMLEFT"
            }
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

    # Obtain item_bboxes and item_local_paths dictionaries.
    item_bboxes, item_local_paths = extract_pics_and_tables_helper(
        pdf_dict=parse_pdf_result_dict,
        base_dir=base_dir,
        output_dir=pics_and_tables_out_path
    )
    
    # Prepare dictionary that will map relative paths of input PDFs with relative paths of output JSONs.
    pdf_processed_json_map = {}

    for pdf_rel_path, input_json_rel_path in tqdm(parse_pdf_result_dict.items(), desc="Processing final JSONs"):
        # Skip already processed PDFs.
        if already_processed and pdf_rel_path in already_processed:
            _log.info(f"Skipping {pdf_rel_path}, already processed.")
            continue
        pdf_path = str(Path(base_dir, pdf_rel_path))
        json_path = str(Path(base_dir, input_json_rel_path))
        # Read intermediate JSON.
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Get file name and its type.
        file_name = data.get("origin", {}).get("filename", "unknown_file")
        file_type = file_name.split(".")[-1] if "." in file_name else "unknown"
        # Define structure of final JSON document.
        output_json = {
            "fileType": file_type,
            "fileName": file_name,
            "fileS3Path": str(Path(pdf_rel_path).relative_to("temp_input_pdfs")),
            # We will add all page data of the file here later.
            "documentPages": []
        }
        # Build reference dictionaries for easier searching within individual groups.
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
        # Dictionary for linking section ID (sectionID) with set of pages that belong to it.
        section_map = {}
        # For storing currently active section.
        current_section = None
        # Dictionary that links section ID with section title.
        section_headers = {}
        # List in which we will store collected data of individual PDF elements.
        final_chunks = []

        # Recursive function that:
        # - Gets reference (e.g. "#/texts/123") and determine if it's text, image, table or group.
        # - If it's 'section_header' create new section.
        # - If it's text ('text' or 'list_item') add data about given PDF element (chunk) to final_chunks;
        #   If it's image or table, add bounding box coordinates and local path to image to the data.
        # - At the end call itself for all children within JSON document structure.
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
            # Skip unused content.
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
            for child_info in node.get("children", []):
                child_ref = child_info.get("$ref", "")
                traverse_node(child_ref)

        # First phase: traverse all children of root 'body' node.
        body = data.get("body", {})
        body_children = body.get("children", [])
        for child in body_children:
            child_ref = child.get("$ref", "")
            traverse_node(child_ref)

        # Second phase: arrange images and tables into sections (if they don't have section assigned already).
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

        # Map sections with lists of pages.
        section_page_map = {}
        for sec_id, pages_set in section_map.items():
            section_page_map[sec_id] = sorted(pages_set)

        # Third phase: fill each chunk or element from final_chunks with content from 'sectionPages'.
        for i, (page_no, chunk) in enumerate(final_chunks):
            sec_id = chunk["sectionID"]
            if sec_id:
                chunk["sectionPages"] = section_page_map.get(sec_id, [])
            else:
                chunk["sectionPages"] = sorted(no_section_pages)

        # Determine sequential position that chunk occupies within entire PDF file and sequential
        # position that it occupies within section.
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

        # Fourth phase: merge data by pages (pageNumber -> list of chunks).
        page_data = {}
        for page_no, chunk in final_chunks:
            if page_no not in page_data:
                page_data[page_no] = {
                    "pageNumber": page_no,
                    "chunks": []
                }
            page_data[page_no]["chunks"].append(chunk)

        # Sort pages and insert them into final dictionary.
        sorted_page_numbers = sorted(page_data.keys())
        output_pages = [page_data[pn] for pn in sorted_page_numbers]
        output_json["documentPages"] = output_pages

        # Construct name of final JSON file.
        json_file_name = os.path.basename(json_path)
        hash_suffix = hash_suffix_map.get(pdf_rel_path, "00000000")
        json_name = Path(json_file_name).stem
        suffix_part = f"_{hash_suffix}.json"
        base_stem = f"{json_name}"
        base_stem_trimmed = base_stem[: max_filename_length - len(suffix_part)]
        output_filename = f"{base_stem_trimmed}{suffix_part}"
        output_json_path = Path(json_out_path) / output_filename

        # Save final dictionary to JSON file.
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        _log.info(f"Processed JSON saved to {output_json_path}")
        pdf_processed_json_map[pdf_rel_path] = str(output_json_path)

    return pdf_processed_json_map