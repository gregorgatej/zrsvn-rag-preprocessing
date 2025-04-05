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

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_pdf(pdf_in_path, json_out_path="intermediate_jsons"):
    pdf_in_path = Path(pdf_in_path)
    json_out_path = Path(json_out_path)
    json_out_path.mkdir(parents=True, exist_ok=True)

    # Setup Docling pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=32, device=AcceleratorDevice.AUTO
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
    )

    result_dict = {}

    pdf_files = (
        [pdf_in_path] if pdf_in_path.is_file()
        else list(pdf_in_path.rglob("*.pdf"))
    )

    for file in tqdm(pdf_files, desc="Parsing PDFs"):
        pdf_dict_key = file.resolve().relative_to(Path.cwd()).as_posix()
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


def extract_data(
    parse_pdf_result_dict,
    base_dir=".",
    pics_and_tables_out_path="output_pics_and_tables",
    json_out_path="output_json"
):
    os.makedirs(pics_and_tables_out_path, exist_ok=True)
    os.makedirs(json_out_path, exist_ok=True)

    hash_suffix_map = {}
    max_filename_length = 150

    def extract_pics_and_tables_helper(pdf_dict, base_dir, output_dir):
        item_bboxes = {}
        def process_items(items, item_type, doc, data, pdf_path, pdf_rel_path, hash_suffix):
            pdf_name = Path(pdf_path).stem
            for idx, item in enumerate(items):
                for prov in item.get("prov", []):
                    page_no = prov["page_no"]
                    page_index = page_no - 1
                    page = doc[page_index]
                    page_width = page.rect.width
                    page_height = page.rect.height
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    all_x = [prov["bbox"]["l"], prov["bbox"]["r"]]
                    all_y = [prov["bbox"]["t"], prov["bbox"]["b"]]
                    for child in item.get("children", []):
                        if "$ref" in child and child["$ref"].startswith("#/texts/"):
                            text_id = int(child["$ref"].split("/")[-1])
                            text_data = data["texts"][text_id]
                            for text_prov in text_data.get("prov", []):
                                if text_prov["page_no"] == page_no:
                                    all_x.extend([text_prov["bbox"]["l"], text_prov["bbox"]["r"]])
                                    all_y.extend([text_prov["bbox"]["t"], text_prov["bbox"]["b"]])
                    l, r = min(all_x), max(all_x)
                    bottom_pdf, top_pdf = min(all_y), max(all_y)
                    dpi_scale = 2
                    l_px = int(l * dpi_scale)
                    r_px = int(r * dpi_scale)
                    t_px = int((page_height - top_pdf) * dpi_scale)
                    b_px = int((page_height - bottom_pdf) * dpi_scale)
                    crop_box = (l_px, t_px, r_px, b_px)
                    if item_type == "picture":
                        suffix_part = f"_pic{idx+1}_pg{page_no}_{hash_suffix}.png"
                    else:
                        suffix_part = f"_tab{idx+1}_pg{page_no}_{hash_suffix}.png"
                    base_stem = pdf_name[: max_filename_length - len(suffix_part)]
                    output_filename = f"{base_stem}{suffix_part}"
                    output_path = os.path.join(output_dir, output_filename)
                    cropped_img = img.crop(crop_box)
                    cropped_img.save(output_path)
                    item_bboxes[(pdf_rel_path, item_type, idx, page_no)] = (l, top_pdf, r, bottom_pdf)
        for pdf_rel_path, json_rel_path in tqdm(pdf_dict.items(), desc="Extracting pics/tables"):
            pdf_path = str(Path(base_dir, pdf_rel_path))
            json_path = str(Path(base_dir, json_rel_path))
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc = fitz.open(pdf_path)
            hash_input = f"{Path(pdf_path).resolve()}_{time.time()}"
            hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            process_items(data.get("pictures", []), "picture", doc, data, pdf_path, pdf_rel_path, hash_suffix)
            process_items(data.get("tables", []), "table", doc, data, pdf_path, pdf_rel_path, hash_suffix)
            hash_suffix_map[pdf_rel_path] = hash_suffix
            doc.close()
        return item_bboxes

    def generate_id():
        return ''.join(random.choices(string.ascii_letters + string.digits, k=9))

    def create_chunk_entry(ref_value, content_type, chunk_id, prov, section_id, section_header, text_value, bounding_box_override=None):
        nr_chars = None
        if content_type == "paragraph":
            charspan = prov.get("charspan", [])
            if len(charspan) > 1:
                nr_chars = charspan[1]
        if bounding_box_override is not None:
            (l_pdf, top_pdf, r_pdf, bottom_pdf) = bounding_box_override
            bounding_box_list = [{
                "l": float(l_pdf),
                "t": float(top_pdf),
                "r": float(r_pdf),
                "b": float(bottom_pdf),
                "coord_origin": "BOTTOMLEFT"
            }]
        else:
            bounding_box_list = [{
                "l": prov["bbox"]["l"],
                "t": prov["bbox"]["t"],
                "r": prov["bbox"]["r"],
                "b": prov["bbox"]["b"],
                "coord_origin": "BOTTOMLEFT"
            }]
        chunk_dict = {
            "ref": ref_value,
            "boundingBox": bounding_box_list,
            "chunkID": chunk_id,
            "contentType": [content_type],
            "sectionPages": [],
            "sectionID": section_id,
            "sectionHeader": section_header,
            "text": text_value,
            "nrCharacters": nr_chars,
            "fileSeqPosition": None,
            "sectionSeqPosition": None
        }
        return chunk_dict

    item_bboxes = extract_pics_and_tables_helper(
        pdf_dict=parse_pdf_result_dict,
        base_dir=base_dir,
        output_dir=pics_and_tables_out_path
    )
    pdf_processed_json_map = {}

    for pdf_rel_path, input_json_rel_path in tqdm(parse_pdf_result_dict.items(), desc="Processing final JSONs"):
        pdf_path = str(Path(base_dir, pdf_rel_path))
        json_path = str(Path(base_dir, input_json_rel_path))
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        file_name = data.get("origin", {}).get("filename", "unknown_file")
        file_type = file_name.split(".")[-1] if "." in file_name else "unknown"
        output_json = {
            "fileType": file_type,
            "fileName": file_name,
            "filePath": pdf_rel_path,
            "documentPages": []
        }
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
        section_map = {}
        current_section = None
        section_headers = {}
        final_chunks = []

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
                    new_chunk = create_chunk_entry(
                        ref_value=ref_value,
                        content_type="picture",
                        chunk_id=chunk_id,
                        prov=prov,
                        section_id=None,
                        section_header=None,
                        text_value=None,
                        bounding_box_override=bbox_override
                    )
                    final_chunks.append((page_no, new_chunk))
            elif label == "table":
                i = int(ref_value.split("/")[-1])
                chunk_id = generate_id()
                for prov in node.get("prov", []):
                    page_no = prov["page_no"]
                    bb_key = (pdf_rel_path, "table", i, page_no)
                    bbox_override = item_bboxes.get(bb_key, None)
                    new_chunk = create_chunk_entry(
                        ref_value=ref_value,
                        content_type="table",
                        chunk_id=chunk_id,
                        prov=prov,
                        section_id=None,
                        section_header=None,
                        text_value=None,
                        bounding_box_override=bbox_override
                    )
                    final_chunks.append((page_no, new_chunk))
            for child_info in node.get("children", []):
                child_ref = child_info.get("$ref", "")
                traverse_node(child_ref)

        body = data.get("body", {})
        body_children = body.get("children", [])
        for child in body_children:
            child_ref = child.get("$ref", "")
            traverse_node(child_ref)

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

        section_page_map = {}
        for sec_id, pages_set in section_map.items():
            section_page_map[sec_id] = sorted(pages_set)

        for i, (page_no, chunk) in enumerate(final_chunks):
            sec_id = chunk["sectionID"]
            if sec_id:
                chunk["sectionPages"] = section_page_map.get(sec_id, [])
            else:
                chunk["sectionPages"] = sorted(no_section_pages)

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

        page_data = {}
        for page_no, chunk in final_chunks:
            if page_no not in page_data:
                page_data[page_no] = {
                    "pageNumber": page_no,
                    "chunks": []
                }
            page_data[page_no]["chunks"].append(chunk)

        sorted_page_numbers = sorted(page_data.keys())
        output_pages = [page_data[pn] for pn in sorted_page_numbers]
        output_json["documentPages"] = output_pages

        json_file_name = os.path.basename(json_path)
        hash_suffix = hash_suffix_map.get(pdf_rel_path, "00000000")
        json_name = Path(json_file_name).stem
        suffix_part = f"_{hash_suffix}.json"
        base_stem = f"{json_name}"
        base_stem_trimmed = base_stem[: max_filename_length - len(suffix_part)]
        output_filename = f"{base_stem_trimmed}{suffix_part}"
        output_json_path = Path(json_out_path) / output_filename

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        _log.info(f"Processed JSON saved to {output_json_path}")
        pdf_processed_json_map[pdf_rel_path] = str(output_json_path)

    return pdf_processed_json_map
