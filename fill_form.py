"""
Fills a PDF form image using a multi-stage approach:
1. Gemini generates a plan mapping data to visual field descriptions.
2. YOLO detects bounding boxes.
3. Gemini (or logic) matches the plan items to detected boxes.
4. Data is drawn onto the image based on the final mapping.
"""

import os
import sys
import argparse
import json # For parsing Gemini responses
from io import BytesIO

import vertexai
from google.cloud import aiplatform
from PIL import Image, ImageDraw, ImageFont
from vertexai.generative_models import (
    GenerationConfig, GenerativeModel,
    HarmBlockThreshold, HarmCategory, Part
)

# --- Add script directory to path if needed ---
# (Keep existing import logic for scripts.inference)
try:
    from scripts.inference import run_inference
except ImportError:
    print("Could not import directly, attempting to add parent dir to path...")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
    try:
        from scripts.inference import run_inference
        print("Import successful after adding path.")
    except ImportError as e:
        print(f"Fatal: Could not import run_inference from scripts.inference. Error: {e}")
        print("Ensure inference.py is in a 'scripts' subdirectory relative to fill_form.py or adjust sys.path.")
        sys.exit(1)

# --- Configuration ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = "us-west4"
# Use a model capable of JSON output and visual analysis
PLAN_MODEL_ID = "gemini-2.0-flash-001" # Or pro
MATCH_MODEL_ID = "gemini-2.0-flash-001" # Can potentially be flash if prompt is good

if not PROJECT_ID:
    raise ValueError("Please set the GOOGLE_CLOUD_PROJECT environment variable.")

# --- Drawing Helper Functions ---

def draw_text_on_image(image, text, position, font, color="black"):
    """Draws text onto a PIL Image object using a pre-loaded font."""
    draw = ImageDraw.Draw(image)
    # Draw the text using the provided font object
    draw.text(position, text, fill=color, font=font)
    return image

def draw_checkbox_on_image(image, center_position, size=10, checked=True, style='fill', color="black", thickness=2):
    """Draws a checkmark or fills a box on a PIL Image object."""
    if not checked:
        return image # Do nothing if not checked
    draw = ImageDraw.Draw(image)
    cx, cy = center_position
    half_size = size / 2
    # Calculate approximate bounding box based on center and size
    x0 = cx - half_size
    y0 = cy - half_size
    x1 = cx + half_size
    y1 = cy + half_size

    if style == 'fill':
        draw.rectangle([x0, y0, x1, y1], fill=color)
    elif style == 'checkmark':
        # Simple '✓' shape - adjust points as needed for aesthetics
        pt1 = (x0 + size * 0.1, y0 + size * 0.5)
        pt2 = (x0 + size * 0.4, y0 + size * 0.8)
        pt3 = (x1 - size * 0.1, y0 + size * 0.2)
        draw.line([pt1, pt2, pt3], fill=color, width=thickness)
    elif style == 'x':
        # Draw two lines for an 'X'
        draw.line([(x0, y0), (x1, y1)], fill=color, width=thickness)
        draw.line([(x0, y1), (x1, y0)], fill=color, width=thickness)
    return image

# --- Image Cropping ---
def crop_with_padding(image, box, padding=30):
    """Crops the image around the box with padding."""
    width, height = image.size
    # Ensure coordinates are within image bounds
    x0 = max(0, box[0] - padding)
    y0 = max(0, box[1] - padding)
    x1 = min(width, box[2] + padding)
    y1 = min(height, box[3] + padding)
    return image.crop((x0, y0, x1, y1))

# --- Gemini Interaction ---
def get_match_from_gemini(model, highlighted_image_bytes, field_type, available_data_items):
    """
    Asks Gemini to match a highlighted field on the full image with available data items.

    Args:
        model: Initialized Vertex AI GenerativeModel.
        highlighted_image_bytes: Bytes of the full form image with one field's bounding box drawn on it.
        field_type: String describing the field type (e.g., 'text_input').
        available_data_items: List of data item dictionaries ({'description': ..., 'value': ...}) currently available.

    Returns:
        The matched data item dictionary from the list, or None if no match found.
    """
    if not available_data_items:
        return None # Nothing to match against

    # No image conversion needed here, bytes are passed in

    # Format available data for the prompt, including index
    options_str = "\\n".join([
        f"{i+1}. Description: {item['description']}, Value: {repr(item['value'])}"
        for i, item in enumerate(available_data_items)
    ])

    # Updated prompt: Refer to the highlighted box on the single image
    prompt = f"""Analyze the provided form image. One field on the form is highlighted with a bounding box.
Focus on the field inside the **highlighted bounding box**.
Field Type Hint for the highlighted field: {field_type}

Available data items (with index numbers):
{options_str}

Which single data item index number (from 1 to {len(available_data_items)}) from the list above is the most appropriate match for the field inside the **highlighted bounding box**?
Consider the visual appearance of the highlighted field, its context within the full form, and the description/value of the data items.

Respond with ONLY the index number of the best matching data item.
If no data item seems appropriate, respond with the exact string 'None'."""

    # Prepare multimodal input (single highlighted image)
    image_part = Part.from_data(data=highlighted_image_bytes, mime_type="image/png")
    contents = [image_part, prompt]

    # Configure safety settings (can reuse existing ones or customize)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    # Configure generation settings (adjust as needed, reduce max tokens for index)
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.8,
        top_k=5,
        candidate_count=1,
        max_output_tokens=10, # Expecting just an index or 'None'
    )

    try:
        response = model.generate_content(
            contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )
        response_text = response.text.strip()
        print(f"    Gemini Match Response for Highlighted Box: {response_text}")

        if response_text.lower() == 'none':
            print("    -> Gemini indicated no match for this highlighted box.")
            return None

        # Attempt to parse the index (1-based from prompt)
        try:
            matched_index_one_based = int(response_text)
            matched_index_zero_based = matched_index_one_based - 1

            # Validate the index
            if 0 <= matched_index_zero_based < len(available_data_items):
                matched_item = available_data_items[matched_index_zero_based]
                print(f"    -> Gemini matched highlighted box to Item Index {matched_index_one_based}: {matched_item['description']}")
                return matched_item # Return the original dictionary
            else:
                print(f"    Error: Gemini returned invalid index {matched_index_one_based} for {len(available_data_items)} available items.")
                return None

        except ValueError:
            print(f"    Error: Could not parse index from Gemini response '{response_text}'. Expected an integer or 'None'.")
            return None

    except Exception as e:
        print(f"    Error calling Gemini API for crop matching: {e}")
        return None

# --- Phase 1: Generate Plan --- (NEW FUNCTION)
def generate_filling_plan(model, form_image_bytes, data_to_fill):
    """Uses Gemini to create a plan mapping data items to form field descriptions."""
    print("--- Phase 1: Generating Filling Plan with Gemini ---")

    # Prepare prompt
    # Updated to handle list of dictionaries with 'description' and 'value'
    data_list_str = "\\n".join([f"- Description: {item['description']}, Value: {repr(item['value'])}" for item in data_to_fill])
    prompt = f"""Analyze the provided form image visually.
Based ONLY on the visual layout and field labels/questions on the form, identify the distinct fillable fields (e.g., 'Full Name text input', '1. Available Nov-Apr 24 (Yes checkbox)', '10. Preferred Mon checkbox', '3. Experience text input', etc.).

Now, look at the provided list of data items, each with a description and a value:
{data_list_str}

Create a plan by matching EACH data item (using its description and value) from the list to EXACTLY ONE of the visually identified fields on the form. Ensure the data type seems appropriate for the field type (e.g., boolean for checkbox, string for text input).

Format the output STRICTLY as a JSON object containing two keys:
1.  `"plan"`: An object where keys are descriptive strings uniquely identifying the field on the form (use question number or nearby text for context), and values are the corresponding *original data item dictionary* (containing both 'description' and 'value') from the list that should go there.
2.  `"unmatched_data"`: A list containing any data item dictionaries from the provided list that could not be confidently matched to a visual field on the form.

Example of desired JSON output format (note the value in the plan is the original dictionary):
```json
{{
  "plan": {{
    "1. Available Nov 1-Apr 24 (Yes checkbox)": {{ "description": "Availability Status Nov-Apr", "value": true }},
    "3. Experience (text input)": {{ "description": "Work Experience Details", "value": "Server at Cafe Bistro..." }},
    "10. Preferred Monday (checkbox)": {{ "description": "Prefers Monday", "value": true }}
  }},
  "unmatched_data": [ {{ "description": "Leftover Data Field", "value": "Some leftover data" }} ]
}}
```

Provide ONLY the JSON object in your response, enclosed in ```json ... ```.
"""

    # Prepare image input
    image_part = Part.from_data(data=form_image_bytes, mime_type="image/png")
    contents = [image_part, prompt]

    # Configure generation for JSON output
    generation_config = GenerationConfig(
        temperature=0.1, # Low temp for deterministic plan
        top_p=0.95,
        top_k=40,
        candidate_count=1,
        max_output_tokens=8192, # Allow enough tokens for potentially large JSON
        # Specify JSON response type (if using a model version that supports it explicitly)
        response_mime_type="application/json", # Uncomment if supported and desired
    )

    # Configure safety settings (optional but recommended)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    # Call Gemini
    try:
        response = model.generate_content(
            contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )

        # --- Parse the Response --- #
        raw_text = response.text
        print(f"Raw Gemini Plan Response:\n{raw_text}")

        # Extract JSON block (handle potential markdown ```json ... ```)
        json_start = raw_text.find("```json")
        json_end = raw_text.rfind("```")

        if json_start != -1 and json_end != -1 and json_start < json_end:
            json_string = raw_text[json_start + 7 : json_end].strip()
        else:
            # Assume the whole response might be JSON if markdown is missing
            json_string = raw_text.strip()

        # Replace Python bools/None with JSON bools/null before parsing
        # Handles booleans/None both as values in key-value pairs and as elements in lists
        replacements = {
            ": True,": ": true,",
            ": False,": ": false,",
            ": None,": ": null,",
            ": True}": ": true}",
            ": False}": ": false}",
            ": None}": ": null}",
            ": True\n": ": true\n",
            ": False\n": ": false\n",
            ": None\n": ": null\n",
            "[True": "[true",
            "[False": "[false",
            "[None": "[null",
            ", True": ", true",
            ", False": ", false",
            ", None": ", null",
            " True]": " true]",
            " False]": " false]",
            " None]": " null]",
            "True,": "true,", # Added for standalone list items
            "False,": "false,", # Added for standalone list items
            "None,": "null," # Added for standalone list items
            # Add more specific cases if needed, but this covers common scenarios
        }
        for py_literal, json_literal in replacements.items():
            json_string = json_string.replace(py_literal, json_literal)

        # Final check for simple True/False at the end of a list item (less common but possible)
        json_string = json_string.replace("True]", "true]")
        json_string = json_string.replace("False]", "false]")
        json_string = json_string.replace("None]", "null]")

        try:
            parsed_result = json.loads(json_string)
            # Basic validation of structure
            if isinstance(parsed_result, dict) and "plan" in parsed_result and "unmatched_data" in parsed_result:
                print("Plan generation and JSON parsing successful.")
                # --- Type Correction Logic - Needs Adaptation ---
                # The plan now holds dictionaries, not just values. We need to ensure these dictionaries
                # are the *original* dictionaries from data_to_fill to preserve types correctly,
                # especially for boolean values that might be stringified/parsed differently.
                plan_dict = parsed_result.get("plan", {})
                unmatched_list = parsed_result.get("unmatched_data", [])
                corrected_plan = {}
                corrected_unmatched = []

                # Create a mapping of string representations of values (or descriptions) for lookup
                original_data_map = { item['description']: item for item in data_to_fill } # Match based on description? Or a combo? Using description for now.

                # Correct the plan items
                for field_desc, planned_item_dict in plan_dict.items():
                    if isinstance(planned_item_dict, dict) and 'description' in planned_item_dict:
                        # Try to find the original dictionary based on description
                        original_item = original_data_map.get(planned_item_dict['description'])
                        if original_item:
                             # Check if value roughly matches (handles potential type changes during JSON conversion)
                             if str(original_item['value']) == str(planned_item_dict.get('value')):
                                corrected_plan[field_desc] = original_item # Use the original item
                             else:
                                print(f"Warning: Value mismatch for description '{planned_item_dict['description']}'. Plan: {planned_item_dict.get('value')}, Original: {original_item['value']}. Using original.")
                                corrected_plan[field_desc] = original_item # Still prefer original structure
                        else:
                            print(f"Warning: Could not find original data item for plan description '{planned_item_dict['description']}'. Using item from plan directly.")
                            corrected_plan[field_desc] = planned_item_dict # Fallback
                    else:
                        print(f"Warning: Invalid item format in plan for field '{field_desc}'. Skipping.")

                # Correct the unmatched items (similar logic)
                for unmatched_item_dict in unmatched_list:
                     if isinstance(unmatched_item_dict, dict) and 'description' in unmatched_item_dict:
                        original_item = original_data_map.get(unmatched_item_dict['description'])
                        if original_item and str(original_item['value']) == str(unmatched_item_dict.get('value')):
                            corrected_unmatched.append(original_item)
                        else:
                             print(f"Warning: Could not reliably match unmatched item '{unmatched_item_dict.get('description')}' back to original data. Adding as is.")
                             corrected_unmatched.append(unmatched_item_dict) # Fallback
                     else:
                         print("Warning: Invalid item format in unmatched_data list. Skipping item.")


                parsed_result["plan"] = corrected_plan
                parsed_result["unmatched_data"] = corrected_unmatched
                return parsed_result
            else:
                print("Error: Parsed JSON does not have the expected structure ('plan' and 'unmatched_data' keys). Returning empty plan.")
                return {"plan": {}, "unmatched_data": data_to_fill}

        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode Gemini response as JSON: {e}. Returning empty plan.")
            print("Response Text was:", json_string)
            return {"plan": {}, "unmatched_data": data_to_fill}

    except Exception as e:
        print(f"Error during Gemini API call for plan generation: {e}")
        return {"plan": {}, "unmatched_data": data_to_fill}

# --- Phase 3: Match Boxes to Data via Highlighted Image ---
def match_plan_to_boxes(model, base_image, plan, detected_boxes):
    """Uses Gemini to match detected boxes (highlighted on full image) to available data items."""
    print("--- Phase 3: Matching Detected Boxes to Data Items via Highlighted Image ---")
    final_mapping = {}
    # Make a mutable list of the data item dictionaries from the plan
    # The plan maps descriptions -> data_dict. We just need the list of data_dicts.
    available_data_items = list(plan.values()) if plan else []

    if not detected_boxes:
        print("Warning: No detected boxes provided. Cannot perform matching.")
        return final_mapping

    if not available_data_items:
        print("Warning: No data items available from the plan. Cannot perform matching.")
        return final_mapping

    print(f"Attempting to match {len(detected_boxes)} boxes to {len(available_data_items)} data items.")

    # Iterate through detected boxes
    for i, box_dict in enumerate(detected_boxes):
        if not available_data_items:
            print("  No more data items available to match. Stopping box iteration.")
            break # Stop if we've used all data

        box_coords = box_dict['box']
        field_type_hint = box_dict.get('label', 'unknown_field') # Use label as hint
        confidence = box_dict.get('confidence', 0)

        print(f"\n  Processing Box {i}: Coords={box_coords}, Type Hint='{field_type_hint}', Conf={confidence:.2f}")

        # --- Create Highlighted Image --- #
        highlighted_image = None # Initialize
        try:
            highlighted_image = base_image.copy()
            draw = ImageDraw.Draw(highlighted_image)
            # Use a distinct color (e.g., magenta) and thickness for the highlight
            highlight_color = "magenta"
            highlight_thickness = 2 # Adjust as needed
            draw.rectangle(box_coords, outline=highlight_color, width=highlight_thickness)

            # Optional: Save highlighted image for debugging
            # highlighted_image.save(f"debug_highlighted_box_{i}.png")

            # Convert highlighted image to bytes for Gemini
            buffer = BytesIO()
            highlighted_image.save(buffer, format="PNG")
            highlighted_image_bytes = buffer.getvalue()

        except Exception as e:
            print(f"    Error creating highlighted image for box {i}: {e}. Skipping box.")
            if highlighted_image: # Clean up potential partial image
                 del highlighted_image
            continue
        # --- End Highlight Creation --- #

        # Ask Gemini to match the highlighted box against the *remaining* available data items
        print(f"    Asking Gemini to match highlighted box against {len(available_data_items)} remaining data items...")
        # Pass the highlighted image bytes
        matched_item = get_match_from_gemini(model, highlighted_image_bytes, field_type_hint, available_data_items)

        # Clean up the copied image object to save memory
        del highlighted_image
        del highlighted_image_bytes # And its bytes

        if matched_item:
            # Match found!
            # Use the description from the matched item as the key in final_mapping
            # This assumes descriptions uniquely identify the data items.
            field_description = matched_item['description']
            if field_description in final_mapping:
                # This situation is less likely now but handle it defensively
                print(f"    Warning: Data item '{field_description}' was already matched. Overwriting previous match for this data item with box {i}.")
            else:
                print(f"    Successfully matched Box {i} to Data: '{field_description}'")

            final_mapping[field_description] = {
                'data': matched_item, # Store the whole original item
                'box': box_dict      # Store the box info associated with this match
            }

            # Remove the matched item from the available list
            try:
                available_data_items.remove(matched_item)
                print(f"    Removed '{field_description}' from available items. {len(available_data_items)} remaining.")
            except ValueError:
                # Should not happen if matched_item came from the list, but handle defensively
                print(f"    Error: Could not remove matched item '{field_description}' from available list.")
        else:
            # No confident match found by Gemini for this highlighted box
            print(f"    -> No data item match found for Box {i} (highlighted).")

    # Report unmatched items
    if available_data_items:
        print(f"\nWarning: {len(available_data_items)} data items could not be matched to any detected box:")
        for item in available_data_items:
            print(f"  - {item['description']}: {item['value']}")
    else:
        print("\nAll data items were successfully matched to boxes.")

    print(f"Highlight-based matching complete. {len(final_mapping)} items mapped.")
    return final_mapping

# --- Phase 4: Draw Final Mapping --- (NEW FUNCTION)
def draw_final_mapping(base_image, final_mapping, font_path, font_size):
    """Draws the data onto the image based on the final box-to-data mapping."""
    print("--- Phase 4: Drawing Final Mapping on Image ---")
    drawn_image = base_image.copy()
    draw = ImageDraw.Draw(drawn_image) # Create Draw object once

    # --- Load Font --- #
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Warning: Font '{font_path}' not found. Using default font.")
        font = ImageFont.load_default()
    # --- Get Font Metrics (assuming fixed font for all text) --- #
    try:
        # Get ascent (height above baseline) and descent (depth below baseline)
        ascent, descent = font.getmetrics()
        text_height_metric = ascent + descent # Total height based on font metrics
        print(f"Using font: {font_path}, Size: {font_size}, Calculated Height (ascent+descent): {text_height_metric}")
    except AttributeError:
        # Fallback for default bitmap font which lacks getmetrics
        print("Warning: Cannot get metrics from default font. Using approximate height.")
        text_height_metric = font_size # Approximate height
    # --- Define Padding --- #
    left_padding = 3  # Pixels from left edge of box
    bottom_padding = 1 # Pixels from bottom edge of box

    if not final_mapping:
        print("Warning: Final mapping is empty. Nothing to draw.")
        return drawn_image

    items_drawn = 0
    for field_description, mapping_info in final_mapping.items():
        try:
            data_dict = mapping_info['data']
            if isinstance(data_dict, dict) and 'value' in data_dict:
                actual_value = data_dict['value']
            else:
                print(f"    Warning: Unexpected data format for field '{field_description}'. Expected dict with 'value', got: {type(data_dict)}. Skipping.")
                continue

            box_dict = mapping_info['box']
            x0, y0, x1, y1 = box_dict['box']
            box_width = x1 - x0
            box_height = y1 - y0

            print(f"  Processing '{field_description}': Value='{actual_value}', Box={box_dict['box']}")

            if isinstance(actual_value, bool):
                # --- Checkbox Drawing --- #
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2
                check_size = max(5, min(box_width, box_height) * 0.7)
                if actual_value:
                    # Pass the pre-created draw object
                    drawn_image = draw_checkbox_on_image(drawn_image, (center_x, center_y), size=check_size, checked=True, style='fill', color='black')
                    print(f"    -> Drawn checkbox/radio at ~({center_x:.0f}, {center_y:.0f})")
                    items_drawn += 1
                else:
                    print(f"    -> Skipped drawing checkbox/radio for False value at ~({center_x:.0f}, {center_y:.0f})")

            elif isinstance(actual_value, (str, int, float)):
                # --- Text Drawing (Bottom-Left Alignment) --- #
                text_to_draw = str(actual_value)
                if not text_to_draw: # Skip empty strings
                     print(f"    -> Skipped drawing empty text.")
                     continue

                # Calculate X position (Left aligned with padding)
                x_pos = x0 + left_padding

                # Calculate Y position (Bottom aligned using font metrics)
                # We want the bottom of the text box (baseline + descent, effectively text_height_metric) 
                # to be at y1 - bottom_padding
                y_pos = y1 - text_height_metric - bottom_padding
                
                # Adjust if calculated position is too high (e.g., tall box, short text)
                y_pos = max(y0, y_pos) # Don't let text start above the box top

                # Pass the pre-loaded font object
                drawn_image = draw_text_on_image(drawn_image, text_to_draw, (x_pos, y_pos), font, color="black")
                print(f"    -> Drawn text '{text_to_draw[:30]}...' starting at ({x_pos:.0f}, {y_pos:.0f})")
                items_drawn += 1
            else:
                print(f"    Warning: Unsupported data type '{type(actual_value)}' for drawing at box {box_dict['box']}. Skipping.")
        except KeyError as e:
             print(f"    Error: Missing key '{e}' while processing field '{field_description}'. Data: {mapping_info}. Skipping.")
        except Exception as e:
            print(f"Error drawing data for field '{field_description}' (Box: {box_dict.get('box', 'N/A')}): {e}")

    print(f"Drawing complete. Attempted to draw {items_drawn} items based on final mapping.")
    return drawn_image

# --- Main Execution --- (Modified)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill a form image using YOLO detection and a Gemini-driven plan.")
    parser.add_argument("--image", required=True, type=str, help="Path to the blank form image file.")
    parser.add_argument("--model", required=True, type=str, help="Path to the trained YOLOv8 model weights (.pt file).")
    parser.add_argument("--output", type=str, default="planned_filled_form.png", help="Path to save the final filled form image.")
    parser.add_argument("--font_path", type=str, default="/Library/Fonts/Arial Unicode.ttf", help="Path to the .ttf font file.")
    parser.add_argument("--font_size", type=int, default=8, help="Font size for drawing text.")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for YOLO detections (0.0 to 1.0).")
    parser.add_argument("--annotated_detect_output", type=str, default="detected_boxes.png", help="Path to save the intermediate image with YOLO boxes drawn.")

    args = parser.parse_args()

    print(f"Initializing Vertex AI for project: {PROJECT_ID} in location: {LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print("Vertex AI initialized.")

    # --- Load Models ---
    # Consider using Pro for planning if Flash struggles with complex instructions/JSON
    plan_model = GenerativeModel(PLAN_MODEL_ID)
    match_model = GenerativeModel(MATCH_MODEL_ID)
    print(f"Using Plan Model: {PLAN_MODEL_ID}, Match Model: {MATCH_MODEL_ID}")

    # --- Load Image Bytes (once) ---
    try:
        print(f"Loading base image: {args.image}")
        base_image = Image.open(args.image).convert("RGB")
        buffer = BytesIO()
        base_image.save(buffer, format="PNG")
        form_image_bytes = buffer.getvalue()
    except FileNotFoundError:
        print(f"Error: Input image file not found at {args.image}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # --- Prepare Data (Using the same fake data for now) ---
    # !!! IMPORTANT: This data should ideally come from a structured source !!!
    # Updated fake data with descriptions
    fake_data = [
        {"description": "Ordering Company Name", "value": "Example Corp"},
        {"description": "Ordering Company Address", "value": "123 Supplier St"},
        {"description": "Ordering Company City", "value": "Supplytown"},
        {"description": "Ordering Company State", "value": "CA"},
        {"description": "Ordering Company Zip Code", "value": "90210"},
        {"description": "Ordering Company Contact Person", "value": "John Doe"},
        {"description": "Ordering Company Phone Number", "value": "555-123-4567"},
        {"description": "Ordering Company Fax Number", "value": "555-987-6543"},
        {"description": "Ordering Company Email Address", "value": "john.doe@example.com"},
        {"description": "Ordering Company Website", "value": "www.example.com"},
        {"description": "Has DUNS Number (Checkbox)", "value": True},
        {"description": "DUNS Number", "value": "123456789"},
        {"description": "Has SIC/NAICS Code (Checkbox)", "value": True},
        {"description": "SIC/NAICS Code", "value": "541511"},
        {"description": "Provider Type: Wholesale (Checkbox)", "value": False},
        {"description": "Provider Type: Retail (Checkbox)", "value": False},
        {"description": "Provider Type: Manufacturing (Checkbox)", "value": True},
        {"description": "Provider Type: Service Provider (Checkbox)", "value": False},
        {"description": "Is Small Business (Yes Option)", "value": True},
        {"description": "Is Small Business (No Option)", "value": False}, # Redundant but for matching
        {"description": "Is Diversity-Owned Business (Yes Option)", "value": False},
        {"description": "Is Diversity-Owned Business (No Option)", "value": True},
        # Link is for registration, not a fillable field
        {"description": "Remittance Payee Name", "value": "Example Corp Remittance"},
        {"description": "Remittance Address", "value": "456 Payment Ave"},
        {"description": "Remittance City", "value": "Payville"},
        {"description": "Remittance State", "value": "NY"},
        {"description": "Remittance Zip Code", "value": "10001"},
        {"description": "Remittance Contact Person", "value": "Accounts Payable"},
        {"description": "Remittance Phone Number", "value": "555-111-2222"},
        {"description": "Remittance Fax Number", "value": "555-333-4444"},
        {"description": "Discount Percentage", "value": "2"}, # Should be number? Using string for now based on prev
        {"description": "Discount Days", "value": "10"}, # Should be number?
        {"description": "Discount Terms: Net (Checkbox)", "value": True},
        # Xcel Energy Policy Net Terms 30 unless discount is given - Not a fillable field
        {"description": "Is Employee of Xcel Energy (Yes Option)", "value": False},
        {"description": "Is Employee of Xcel Energy (No Option)", "value": True},
        {"description": "Related to Employee of Xcel Energy (Yes Option)", "value": False},
        {"description": "Related to Employee of Xcel Energy (No Option)", "value": True},
        # W9 Information Section Header - Not a fillable field
        {"description": "W9 Checkbox: Individual/Sole Proprietor", "value": True},
        {"description": "W9 Checkbox: Corporation", "value": False},
        {"description": "W9 Checkbox: Partnership", "value": False},
        {"description": "W9 Checkbox: Other", "value": False},
        {"description": "State of Incorporation", "value": "Delaware"},
        {"description": "Exempt from Backup Withholding (Yes Option)", "value": False},
        {"description": "Exempt from Backup Withholding (No Option)", "value": True},
        {"description": "Federal Tax ID Number", "value": "12-3456789"},
        {"description": "Tax ID Type: Manufacturer (Radio)", "value": False},
        {"description": "Tax ID Type: Vendor (Radio)", "value": False},
        {"description": "Tax ID Type: Both (Radio)", "value": True},
        # Social Security # (if applicable) - Assuming Federal Tax ID is provided, maybe add placeholder? {"description": "Social Security Number (Optional)", "value": ""},
        {"description": "IRS Registered Name", "value": "Example Corp"},
        {"description": "IRS Registered Address", "value": "123 Tax St"},
        {"description": "IRS Registered City", "value": "Taxington"},
        {"description": "IRS Registered State", "value": "DE"},
        {"description": "IRS Registered Zip Code", "value": "19901"},
        # Signature field - Usually not filled by script unless generating signature image
        # Date field - Often needs current date logic
        {"description": "Form Signature Date", "value": "2023-10-27"}
    ]

    # --- Phase 1: Generate Plan ---
    plan_result = generate_filling_plan(plan_model, form_image_bytes, fake_data)
    filling_plan = plan_result.get("plan", {})
    unmatched_data = plan_result.get("unmatched_data", [])

    if not filling_plan:
        print("Error: Gemini failed to generate a filling plan. Exiting.")
        sys.exit(1)
    print(f"Generated plan with {len(filling_plan)} items.")
    if unmatched_data:
        print(f"Warning: {len(unmatched_data)} data items could not be matched to form fields by Gemini: {unmatched_data}")

    # --- Phase 2: Run YOLO Inference ---
    print("--- Phase 2: Running Bounding Box Detection ---")
    detected_bboxes = run_inference(
        model_path=args.model,
        image_path=args.image,
        output_path=args.annotated_detect_output,
        conf_thresh=args.conf
    )
    if not detected_bboxes:
        print("Warning: No bounding boxes were detected by the YOLO model.")
        # Continue? Or exit? For now, continue, but matching will fail.

    print(f"Detected {len(detected_bboxes)} boxes above threshold {args.conf}.")

    # --- Phase 3: Match Plan to Boxes ---
    final_mapping = match_plan_to_boxes(match_model, base_image, filling_plan, detected_bboxes)

    if not final_mapping:
        print("Warning: Failed to match any planned items to detected boxes.")
        # Decide if we should still save an image (maybe just the original?) - saving drawn for now

    # --- Phase 4: Draw Final Mapping ---
    final_image = draw_final_mapping(base_image, final_mapping, args.font_path, args.font_size)

    # --- Save Final Output ---
    try:
        print(f"Saving final filled image to: {args.output}")
        final_image.save(args.output)
        print("Final image saved successfully.")
    except Exception as e:
        print(f"Error saving final image: {e}")

    print("Script finished.") 