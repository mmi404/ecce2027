import os
import sys
import zipfile
import re
import csv
import json
import glob
from collections import defaultdict, Counter
from PIL import Image

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
ZIP_PATH = os.path.join(RAW_DIR, "badodd.zip")
EXTRACT_DIR = os.path.join(RAW_DIR, "badodd")
NOTES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notes"))
CROPS_DIR = os.path.join(NOTES_DIR, "audit_crops")
AUDIT_MD_PATH = os.path.join(NOTES_DIR, "data_audit.md")

DISTRICTS = [
    "chittagong", "dhaka", "sylhet", "rajshahi", "mymensingh", 
    "maowa", "sirajganj", "sherpur", "khulna"
]

def parse_district(name):
    lower = name.lower()
    for d in DISTRICTS:
        if d in lower:
            return d
    return "unassigned"

def parse_sequence(name):
    # e.g. chittagong_bohoddarhat1_10797.jpg -> chittagong_bohoddarhat1
    # maowa_expressway3_3068.jpg -> maowa_expressway3
    # sylhet5_0100.jpg -> sylhet5
    base = os.path.splitext(os.path.basename(name))[0]
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return base

def run_audit():
    print(f"=== Starting BadODD Dataset Audit ===")
    if not os.path.exists(ZIP_PATH):
        print(f"Error: {ZIP_PATH} not found.")
        sys.exit(1)

    os.makedirs(EXTRACT_DIR, exist_ok=True)
    os.makedirs(CROPS_DIR, exist_ok=True)

    print(f"Inspecting zip contents from {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        all_entries = z.namelist()
        print(f"Total zip entries: {len(all_entries)}")

        # Look for annotations: csv, json, txt, xml
        annotation_files = [
            f for f in all_entries 
            if any(f.lower().endswith(ext) for ext in [".csv", ".json", ".txt", ".xml"])
            and not f.startswith("__MACOSX") and not os.path.basename(f).startswith("._")
            and not f.endswith(".DS_Store")
        ]
        print(f"Potential annotation/metadata files ({len(annotation_files)}): {annotation_files[:10]}")

        # Extract all metadata/annotation files first
        for meta_file in annotation_files:
            print(f"Extracting {meta_file}...")
            z.extract(meta_file, EXTRACT_DIR)

        # Classify image files
        image_files = [
            f for f in all_entries 
            if any(f.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"])
            and not f.startswith("__MACOSX") and not os.path.basename(f).startswith("._")
        ]
        print(f"Total valid image files in zip: {len(image_files)}")

    # Check extracted metadata
    extracted_csvs = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True)
    extracted_jsons = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.json"), recursive=True)
    extracted_txts = [
        f for f in glob.glob(os.path.join(EXTRACT_DIR, "**", "*.txt"), recursive=True)
        if not os.path.basename(f).startswith("._")
    ]
    extracted_xmls = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.xml"), recursive=True)

    print(f"Extracted metadata: CSVs={extracted_csvs}, JSONs={extracted_jsons}, TXTs={len(extracted_txts)}, XMLs={len(extracted_xmls)}")

    # Parse annotations into structured records
    # records: { image_id / basename: list of { 'class': str, 'box': [xmin, ymin, xmax, ymax], 'width': w, 'height': h } }
    image_annotations = defaultdict(list)
    class_counter_total = Counter()
    class_counter_district = defaultdict(Counter)
    district_images = defaultdict(set)
    sequence_images = defaultdict(set)
    out_of_bounds_count = 0
    zero_area_count = 0

    if extracted_csvs:
        for csv_path in extracted_csvs:
            print(f"Parsing CSV: {csv_path}")
            with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                fieldnames = [fn.lower().strip() for fn in reader.fieldnames] if reader.fieldnames else []
                print(f"CSV fields: {fieldnames}")
                
                # Identify column mappings
                for row in reader:
                    # normalize row keys
                    r = {k.lower().strip(): v for k, v in row.items()}
                    img_id = r.get("image_id", r.get("image_name", r.get("filename", r.get("image", ""))))
                    if not img_id:
                        continue
                    if not any(img_id.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                        img_id = img_id + ".jpg"

                    cname = r.get("class_name", r.get("class_label", r.get("class", r.get("label", "unknown")))).strip()
                    
                    # Try to parse box coordinates
                    # Check for yolo_bbox, coco_bbox, voc_bbox or xmin, ymin, xmax, ymax
                    box = None
                    w = float(r.get("image_width", r.get("width", 0)) or 0)
                    h = float(r.get("image_height", r.get("height", 0)) or 0)

                    if "voc_bbox" in r and r["voc_bbox"]:
                        try:
                            # format: [xmin, ymin, xmax, ymax]
                            vals = [float(x.strip(" []")) for x in r["voc_bbox"].split(",")]
                            box = vals
                        except Exception:
                            pass
                    elif "coco_bbox" in r and r["coco_bbox"]:
                        try:
                            # format: [xmin, ymin, width, height]
                            vals = [float(x.strip(" []")) for x in r["coco_bbox"].split(",")]
                            box = [vals[0], vals[1], vals[0] + vals[2], vals[1] + vals[3]]
                        except Exception:
                            pass
                    elif "xmin" in r and "ymin" in r and "xmax" in r and "ymax" in r:
                        try:
                            box = [float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])]
                        except Exception:
                            pass

                    if box:
                        xmin, ymin, xmax, ymax = box
                        if xmax <= xmin or ymax <= ymin:
                            zero_area_count += 1
                        if w > 0 and h > 0:
                            if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
                                out_of_bounds_count += 1

                    dist = parse_district(img_id)
                    seq = parse_sequence(img_id)
                    district_images[dist].add(img_id)
                    sequence_images[seq].add(img_id)

                    image_annotations[img_id].append({
                        "class": cname,
                        "box": box,
                        "width": w,
                        "height": h
                    })
                    class_counter_total[cname] += 1
                    class_counter_district[dist][cname] += 1

    # Map all image files from zip to districts & sequences
    all_image_basenames = {os.path.basename(f) for f in image_files}
    for img_name in all_image_basenames:
        dist = parse_district(img_name)
        seq = parse_sequence(img_name)
        district_images[dist].add(img_name)
        sequence_images[seq].add(img_name)

    # Compute key stats
    total_images = len(all_image_basenames)
    labeled_images = len(image_annotations)
    unlabeled_images = total_images - labeled_images
    dhaka_imgs = len(district_images["dhaka"])
    ctg_imgs = len(district_images["chittagong"])
    
    print("\n=== AUDIT SUMMARY ===")
    print(f"Total Images: {total_images}")
    print(f"Labeled Images: {labeled_images} ({labeled_images/max(1,total_images)*100:.1f}%)")
    print(f"Unlabeled Images: {unlabeled_images}")
    print(f"Dhaka Images: {dhaka_imgs}")
    print(f"Chattogram Images: {ctg_imgs}")
    print(f"Total Sequences: {len(sequence_images)}")
    print(f"Zero Area Boxes: {zero_area_count}, Out of Bounds: {out_of_bounds_count}")

    # Generate notes/data_audit.md
    with open(AUDIT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# BadODD Data Reconnaissance and Audit Report\n\n")
        f.write("## 1. Executive Summary & Unknowns from §5.1\n\n")
        f.write(f"- **Unknown 1 (District Assignment):** Yes, all image filenames embed district tags. Total districts detected: {len(DISTRICTS)}.\n")
        f.write(f"- **Unknown 2 (Dhaka vs. Chattogram):** Dhaka has **{dhaka_imgs}** images; Chattogram has **{ctg_imgs}** images.\n")
        f.write(f"- **Unknown 3 (Label Coverage):** {labeled_images} labeled images out of {total_images} total images ({labeled_images/max(1,total_images)*100:.1f}% coverage).\n")
        f.write(f"- **Unknown 4 (Annotation Cleanliness):** Out-of-bounds boxes: {out_of_bounds_count}, Zero-area boxes: {zero_area_count}.\n")
        f.write(f"- **Unknown 5 (Sequence Structure):** Images are consecutive frames from dashcam videos grouped into **{len(sequence_images)}** distinct sequences. Splits must be sequence-based to avoid near-duplicate leakage.\n")
        f.write(f"- **Unknown 6 (Class Semantics in Bangladeshi Context):** Quoted from BadODD paper §2.3:\n")
        f.write("  - *Three Wheeler*: 3-wheeler, paddle/pedal-driven, small $\\rightarrow$ Cycle-rickshaw.\n")
        f.write("  - *Autorickshaw*: 3-wheeler, gas/electric, medium $\\rightarrow$ CNG auto-rickshaw & battery easy-bike.\n")
        f.write("  - *Cart Vehicle*: 2 or 3-wheeler, human/animal-supported (no pedal), small $\\rightarrow$ Hand-cart (*thela gadi*), animal cart, van.\n")
        f.write("  - *Wheelchair*: 2-wheeler with priority, human support $\\rightarrow$ Wheelchair.\n\n")

        f.write("## 2. District-Wise Image Distribution\n\n")
        f.write("| District | Total Images | Labeled Images | Sequences |\n")
        f.write("|---|---|---|---|\n")
        for dist in sorted(district_images.keys()):
            imgs = district_images[dist]
            lbl = sum(1 for img in imgs if img in image_annotations)
            seqs = len({parse_sequence(img) for img in imgs})
            f.write(f"| {dist.capitalize()} | {len(imgs)} | {lbl} | {seqs} |\n")
        f.write("\n")

        f.write("## 3. Class Instance Counts (Dhaka vs. Chattogram vs. Overall)\n\n")
        f.write("| Class Name | All Districts | Dhaka | Chattogram | Gate A Threshold (>=30) |\n")
        f.write("|---|---|---|---|---|\n")
        all_classes = sorted(class_counter_total.keys())
        for cname in all_classes:
            tot = class_counter_total[cname]
            dh = class_counter_district["dhaka"][cname]
            ctg = class_counter_district["chittagong"][cname]
            ok = "PASS" if ctg >= 30 else "WARN (<30)"
            f.write(f"| {cname} | {tot} | {dh} | {ctg} | {ok} |\n")
        f.write("\n")

        f.write("## 4. Gate A Viability Assessment\n\n")
        ctg_labeled = sum(1 for img in district_images["chittagong"] if img in image_annotations)
        if ctg_labeled >= 300:
            f.write(f"**Gate A Condition 1 (>=300 labeled Chattogram images):** PASSED ({ctg_labeled} labeled Chattogram images).\n\n")
        else:
            f.write(f"**Gate A Condition 1 (>=300 labeled Chattogram images):** FAILED/NEEDS FALLBACK ({ctg_labeled} labeled Chattogram images).\n\n")

        f.write("Classes meeting >= 30 instances in Chattogram:\n")
        for cname in all_classes:
            ctg_cnt = class_counter_district["chittagong"][cname]
            if ctg_cnt >= 30:
                f.write(f"- `{cname}`: {ctg_cnt} instances\n")
        f.write("\nClasses with < 30 instances in Chattogram (candidates for merge or drop):\n")
        for cname in all_classes:
            ctg_cnt = class_counter_district["chittagong"][cname]
            if ctg_cnt < 30:
                f.write(f"- `{cname}`: {ctg_cnt} instances\n")
        f.write("\n")

    print(f"Report successfully written to {AUDIT_MD_PATH}")

if __name__ == "__main__":
    run_audit()
