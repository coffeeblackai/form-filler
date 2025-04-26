#!/usr/bin/env python3
import os
import shutil
import random
import hashlib
import yaml
try:
    import tqdm
    import fitz  # PyMuPDF
except ImportError:
    print("Error: Required libraries (tqdm, PyMuPDF) not found.")
    print("Please install them: pip install tqdm PyMuPDF")
    exit(1)

# --- Configuration ---
RAW_PDF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/IRS-PDFs'))
# Define base raw directory
RAW_DIR_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw'))
# List of subdirectories within RAW_DIR_BASE containing PDFs to process
PDF_SOURCE_DIRS = [
    "IRS-PDFs",
    "OPM-Standard",
    "OPM-Optional",
    "OPM-OPM",
    "OPM-Retirement",
    "OPM-Investigation",
    "OPM-FEGLI"
]

FINAL_YOLO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/yolo'))
TRAIN_RATIO = 0.8 # 80% for training, 20% for validation
SEED = 42 # For reproducible splits
IMAGE_DPI = 150 # Resolution for rendering PDF pages to images
IMAGE_FORMAT = "png" # Output image format

# Define final output directories
TRAIN_IMG_DIR = os.path.join(FINAL_YOLO_DIR, 'images', 'train')
VAL_IMG_DIR = os.path.join(FINAL_YOLO_DIR, 'images', 'val')
TRAIN_LABEL_DIR = os.path.join(FINAL_YOLO_DIR, 'labels', 'train')
VAL_LABEL_DIR = os.path.join(FINAL_YOLO_DIR, 'labels', 'val')

# --- Class Definition for Interactive Fields ---
# Define classes based on AcroForm field types (/FT)
FIELD_CLASSES = [
    "text_input",  # /Tx
    "checkbox",    # /Btn (when V indicates checkbox style)
    "radio_button",# /Btn (when V indicates radio style)
    "dropdown",    # /Ch
    "signature",   # /Sig
    "button",      # /Btn (generic button, maybe map to other?)
    "other_field"  # Catch-all for unexpected types
]
FIELD_CLASS_MAP = {name: i for i, name in enumerate(FIELD_CLASSES)}
print(f"Using Field Classes: {FIELD_CLASS_MAP}")

# Map PDF field types (/FT) to our class names
# These come from fitz widget.field_type_string now
PDF_FT_TO_CLASS_NAME = {
    "text": "text_input",      # Type 4
    "checkbox": "checkbox",    # Type 2
    "radiobutton": "radio_button", # Type 2
    "combobox": "dropdown",    # Type 1
    "listbox": "dropdown",     # Type 1? Treat as dropdown for now
    "signature": "signature",  # Type 7
    "button": "button",        # Type 0 (Push button) - Map to generic button
    # Others exist but are less common for forms (strikeout, highlight, etc.)
}

# --- Helper Functions ---

def get_split_from_filename(filename, train_ratio, seed):
    """Determines train/val split based on filename hash for consistency."""
    # Use SHA1 hash of filename for deterministic split
    hasher = hashlib.sha1(filename.encode('utf-8'))
    # Use first few bytes of hash and scale to [0,1]
    hash_val = int.from_bytes(hasher.digest()[:4], 'little') / (2**32 - 1)
    # Combine with seed for slight variation if needed across runs
    # Not strictly necessary if filename list is stable
    # random.seed(seed + filename) # Re-seeding can be slow
    # hash_val = (hash_val + random.random()) / 2.0 # Alternative mixing
    
    return 'train' if hash_val < train_ratio else 'val'

# --- Main Processing Function ---

def extract_fields_and_render():
    print("--- Extracting Widget Annotations from PDFs and Creating YOLO Dataset ---")

    # Create final directories
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
    os.makedirs(VAL_LABEL_DIR, exist_ok=True)

    all_pdf_files_map = {} # Store {source_subdir: [list of pdf paths]}
    total_pdf_count = 0
    print(f"Scanning for PDFs in subdirectories of: {RAW_DIR_BASE}")
    for subdir in PDF_SOURCE_DIRS:
        current_dir = os.path.join(RAW_DIR_BASE, subdir)
        if not os.path.isdir(current_dir):
            print(f"Info: Directory not found, skipping: {current_dir}")
            continue
        
        pdf_files_in_subdir = [
            os.path.join(current_dir, f) 
            for f in os.listdir(current_dir) 
            if f.lower().endswith('.pdf')
        ]
        
        if pdf_files_in_subdir:
            all_pdf_files_map[subdir] = pdf_files_in_subdir
            print(f"  Found {len(pdf_files_in_subdir)} PDFs in {subdir}")
            total_pdf_count += len(pdf_files_in_subdir)
        else:
             print(f"  Found 0 PDFs in {subdir}")

    print(f"Total PDFs found across all sources: {total_pdf_count}")
    if total_pdf_count == 0:
        print("No PDF files found in any specified source directory. Exiting.")
        return

    processed_pages_count = 0
    processed_fields_count = 0
    skipped_pdfs_count = 0
    skipped_pages_count = 0

    # Determine split for each PDF based on its filename relative to its source subdir
    # Use a unique identifier (like subdir + filename) for deterministic split
    pdf_splits = {}
    all_pdf_items_for_tqdm = []
    for subdir, pdf_paths in all_pdf_files_map.items():
         for pdf_path in pdf_paths:
              pdf_filename = os.path.basename(pdf_path)
              unique_id = f"{subdir}/{pdf_filename}" # Use subdir/filename for split hash
              split = get_split_from_filename(unique_id, TRAIN_RATIO, SEED)
              pdf_splits[pdf_path] = split
              all_pdf_items_for_tqdm.append((subdir, pdf_path))

    # Process each PDF path collected
    for subdir, pdf_path in tqdm.tqdm(all_pdf_items_for_tqdm, desc="Processing PDFs"):
        split = pdf_splits[pdf_path]
        pdf_filename = os.path.basename(pdf_path)
        pdf_base_name = os.path.splitext(pdf_filename)[0]
        source_prefix = subdir.replace('-','_').lower()

        doc = None
        try:
            doc = fitz.open(pdf_path)
            num_pdf_pages = len(doc)
            if num_pdf_pages == 0:
                skipped_pdfs_count += 1
                continue

            pdf_has_widgets = False
            for page_index in range(num_pdf_pages):
                page = doc.load_page(page_index)
                widgets_on_page = list(page.widgets())
                if not widgets_on_page: continue
                pdf_has_widgets = True

                # --- Get Transformation Matrix & Render --- 
                # Matrix to convert PDF points -> default device pixels (usually 72 DPI, origin top-left)
                try:
                    pdf_to_device_matrix = page.transformation_matrix
                    # Matrix for scaling from default device pixels -> our target DPI
                    dpi_scale = IMAGE_DPI / 72.0
                    render_scale_matrix = fitz.Matrix(dpi_scale, dpi_scale)
                    # Combined matrix: PDF points -> Target DPI pixels (origin top-left)
                    pdf_to_render_matrix = pdf_to_device_matrix * render_scale_matrix
                except Exception as matrix_err:
                     print(f"Warning: Could not get transformation matrix for page {page_index} of {pdf_filename}: {matrix_err}. Skipping page.")
                     skipped_pages_count += 1
                     continue
                 
                # Render the pixmap at target DPI
                try:
                    pix = page.get_pixmap(dpi=IMAGE_DPI)
                    img_width_px, img_height_px = pix.width, pix.height
                    if img_width_px <= 0 or img_height_px <= 0:
                         raise ValueError("Rendered image has zero dimension")
                except Exception as render_err:
                     print(f"Warning: Could not render page {page_index} of {pdf_filename} at DPI {IMAGE_DPI}: {render_err}. Skipping page.")
                     skipped_pages_count += 1
                     continue

                # --- Prepare Output Paths --- 
                page_base_name = f"{source_prefix}_{pdf_base_name}_page_{page_index + 1}"
                img_filename = f"{page_base_name}.{IMAGE_FORMAT}"
                label_filename = f"{page_base_name}.txt"
                img_out_dir = TRAIN_IMG_DIR if split == 'train' else VAL_IMG_DIR
                label_out_dir = TRAIN_LABEL_DIR if split == 'train' else VAL_LABEL_DIR
                img_out_path = os.path.join(img_out_dir, img_filename)
                label_out_path = os.path.join(label_out_dir, label_filename)

                # DEBUG: Check existence
                # print(f"DEBUG: Checking existence for: {img_out_path} and {label_out_path}")
                if os.path.exists(img_out_path) and os.path.exists(label_out_path):
                    # print(f"DEBUG: Files exist, skipping page {page_index+1} of {pdf_filename}")
                    processed_pages_count += 1 # Assume it was processed correctly before
                    continue

                # --- Save Image --- 
                try:
                    # print(f"DEBUG: Attempting to save image: {img_out_path}") # Verbose
                    if not os.path.exists(img_out_path):
                        pix.save(img_out_path, output=IMAGE_FORMAT)
                        # print(f"DEBUG: Saved image: {img_out_path}") # Verbose
                except Exception as img_save_err:
                    print(f"Error saving image for page {page_index} of {pdf_filename}: {img_save_err}")
                    skipped_pages_count += 1
                    continue # Skip to next page
                
                # --- Process Widgets and Create Labels --- 
                yolo_lines = []
                page_field_count = 0
                widgets_processed_count = 0
                widgets_failed_transform_count = 0
                
                for widget in widgets_on_page:
                    widgets_processed_count += 1
                    widget_rect_pdf = widget.rect # fitz.Rect in PDF points

                    # --- Transform Coordinates using Matrix --- 
                    try:
                        # Use top-left and bottom-right points for transformation
                        tl_pdf = widget_rect_pdf.tl
                        br_pdf = widget_rect_pdf.br

                        # Transform points to rendered pixel coordinates (origin top-left)
                        tl_px_tl = tl_pdf * pdf_to_render_matrix
                        br_px_tl = br_pdf * pdf_to_render_matrix

                        # Extract raw pixel coordinates from transformed points
                        px0 = tl_px_tl.x 
                        py0 = tl_px_tl.y 
                        px1 = br_px_tl.x
                        py1 = br_px_tl.y 

                        # Ensure correct ordering (x0 < x1, y0 < y1 for top-left system)
                        x0_px_tl = min(px0, px1)
                        y0_px_tl = min(py0, py1)
                        x1_px_tl = max(px0, px1)
                        y1_px_tl = max(py0, py1)

                        # Calculate center, width, height in pixels - Height MUST be positive now
                        box_width_px = x1_px_tl - x0_px_tl
                        box_height_px = y1_px_tl - y0_px_tl 
                        
                        center_x_px = x0_px_tl + box_width_px / 2.0
                        center_y_px = y0_px_tl + box_height_px / 2.0 

                        # +++ DEBUG: Print intermediate pixel values (UNCOMMENTED) +++
                        if widgets_processed_count <= 5:
                           print(f"  DEBUG Widget {widgets_processed_count}: PDF Rect={widget_rect_pdf}")
                           print(f"  DEBUG Widget {widgets_processed_count}: Raw Pix (px0,py0,px1,py1)=({px0:.2f}, {py0:.2f}, {px1:.2f}, {py1:.2f})") # Print raw before ordering
                           print(f"  DEBUG Widget {widgets_processed_count}: Ordered Pix (x0,y0,x1,y1)=({x0_px_tl:.2f}, {y0_px_tl:.2f}, {x1_px_tl:.2f}, {y1_px_tl:.2f})")
                           print(f"  DEBUG Widget {widgets_processed_count}: Pixel Ctr=({center_x_px:.2f}, {center_y_px:.2f}) W/H=({box_width_px:.2f}, {box_height_px:.2f}) | Img Dims=({img_width_px}x{img_height_px})")
                        # +++ END DEBUG +++

                        # Normalize
                        norm_center_x = max(0.0, min(1.0, center_x_px / img_width_px))
                        norm_center_y = max(0.0, min(1.0, center_y_px / img_height_px))
                        norm_width = max(0.0, min(1.0, box_width_px / img_width_px))
                        norm_height = max(0.0, min(1.0, box_height_px / img_height_px))

                        # --- APPLY Y-FLIP to normalized center Y --- 
                        norm_center_y = 1.0 - norm_center_y
                        # --- END Y-FLIP --- 

                        # Final sanity check for zero size after normalization/float error
                        if norm_width <= 1e-6 or norm_height <= 1e-6: 
                            if widgets_processed_count <= 5: 
                                print(f"  DEBUG Widget {widgets_processed_count}: SKIPPED FINAL CHECK zero norm W/H ({norm_width:.6f}, {norm_height:.6f})")
                            widgets_failed_transform_count += 1 # Count as failure if final size is zero
                            continue

                        yolo_coords = (norm_center_x, norm_center_y, norm_width, norm_height)
                    except Exception as coord_err:
                        # print(f"DEBUG: Coord transform failed for widget on page {page_index+1} of {pdf_filename}") # Verbose
                        widgets_failed_transform_count += 1
                        continue
                    # --- End Transformation --- 

                    field_type_str = widget.field_type_string
                    # Map fitz field type string to our class name
                    class_name = PDF_FT_TO_CLASS_NAME.get(field_type_str.lower()) 

                    # Refine /Btn classification based on flags (example)
                    if field_type_str.lower() == "radiobutton":
                        class_name = "radio_button"
                    elif field_type_str.lower() == "checkbox":
                         class_name = "checkbox"
                    elif field_type_str.lower() == "button": # Pushbutton
                         class_name = "button"
                         
                    # Fallback for unmapped types
                    if class_name is None:
                        # print(f"Info: Unmapped field type '{field_type_str}' on page {page_index} of {pdf_filename}. Using 'other_field'.")
                        class_name = "other_field"

                    if class_name in FIELD_CLASS_MAP:
                        class_id = FIELD_CLASS_MAP[class_name]
                    else:
                        # Should not happen if fallback works, but just in case
                        print(f"Warning: Class name '{class_name}' not in FIELD_CLASS_MAP. Using 'other_field'.")
                        class_id = FIELD_CLASS_MAP["other_field"]

                    if yolo_coords:
                        yolo_lines.append(f"{class_id} {' '.join(map(lambda x: f'{x:.6f}', yolo_coords))}")
                        page_field_count += 1
                
                # Print transform summary per page if issues detected
                # if widgets_failed_transform_count > 0:
                #    print(f"DEBUG: Page {page_index+1} ({pdf_filename}): Processed {widgets_processed_count} widgets, {widgets_failed_transform_count} failed transform, {page_field_count} valid boxes.")

                # --- Write Label File --- 
                if page_field_count > 0:
                    try:
                        # print(f"DEBUG: Attempting to write label: {label_out_path}") # Verbose
                        with open(label_out_path, 'w') as f:
                            f.write("\n".join(yolo_lines))
                        # print(f"DEBUG: Wrote label: {label_out_path}") # Verbose
                        processed_pages_count += 1
                        processed_fields_count += page_field_count
                    except Exception as label_write_err:
                        print(f"Error writing label file {label_out_path}: {label_write_err}")
                        skipped_pages_count += 1
                        # Clean up image if label failed
                        if os.path.exists(img_out_path): 
                            print(f"DEBUG: Deleting image {img_out_path} because label write failed.")
                            try: os.remove(img_out_path)
                            except OSError as del_err: print(f"DEBUG: Failed to delete {img_out_path}: {del_err}")
                else:
                     # No valid fields extracted for this page, clean up image
                     if os.path.exists(img_out_path): 
                        print(f"DEBUG: Deleting image {img_out_path} because no valid fields were extracted after transform (Processed: {widgets_processed_count}, Failed: {widgets_failed_transform_count}, Valid: {page_field_count}).")
                        try: os.remove(img_out_path)
                        except OSError as del_err: print(f"DEBUG: Failed to delete {img_out_path}: {del_err}")
                     # Don't count as a fully processed page if no labels written
                     # Note: We might still count the PDF as processed if other pages worked

            # End of page loop for this PDF
            if not pdf_has_widgets:
                 skipped_pdfs_count += 1 # Count PDF as skipped if no widgets found on *any* page
                 
        except Exception as pdf_process_err:
            print(f"Critical error processing PDF {pdf_filename} from {subdir}: {pdf_process_err}")
            skipped_pdfs_count += 1
            # Fallthrough to finally for cleanup
        finally:
            if doc:
                try: 
                    doc.close()
                except Exception: 
                    pass # Ignore close error

    print("\n--- Processing Summary ---")
    print(f"\nTotal PDFs scanned: {total_pdf_count}")
    print(f"PDFs skipped (no widgets or critical error): {skipped_pdfs_count}")
    print(f"Total Pages Rendered & Processed: {processed_pages_count}")
    print(f"Pages skipped due to errors: {skipped_pages_count}")
    print(f"Total Widget Fields Extracted: {processed_fields_count}")

    # --- Create dataset.yaml --- 
    yaml_path = os.path.join(FINAL_YOLO_DIR, 'dataset.yaml')
    yaml_data = {
        'path': os.path.abspath(FINAL_YOLO_DIR),
        'train': os.path.relpath(TRAIN_IMG_DIR, FINAL_YOLO_DIR),
        'val': os.path.relpath(VAL_IMG_DIR, FINAL_YOLO_DIR),
        'names': {i: name for i, name in enumerate(FIELD_CLASSES)}
    }

    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
        print(f"\nSuccessfully created dataset configuration: {yaml_path}")
        print(f"Dataset ready in: {FINAL_YOLO_DIR}")
    except Exception as e:
        print(f"Error creating dataset.yaml file at {yaml_path}: {e}")

if __name__ == "__main__":
    extract_fields_and_render() 