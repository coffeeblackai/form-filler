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

def draw_text_on_image(image, text, position, font_path="arial.ttf", font_size=12, color="black"):
    """Draws text onto a PIL Image object."""
    draw = ImageDraw.Draw(image)
    try:
        # Load the font
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Warning: Font '{font_path}' not found. Using default font.")
        # Fallback to Pillow's default bitmap font if the TrueType font fails
        font = ImageFont.load_default()
    # Draw the text
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
def get_match_from_gemini(model, image_crop, field_type, available_data_list):
    """
    Asks Gemini to match a field crop with available data.

    Args:
        model: Initialized Vertex AI GenerativeModel.
        image_crop: PIL Image object of the cropped field area.
        field_type: String describing the field type (e.g., 'text_input').
        available_data_list: List of data values currently available.

    Returns:
        The matched data value from the list, or None if no match found.
    """
    # Convert PIL Image to bytes
    buffer = BytesIO()
    image_crop.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    # Corrected prompt f-string
    prompt = f"""Analyze the following form field image crop.
Field Type Hint: {field_type}

Available data options:
{available_data_list}

Which single data value from the list is the most appropriate match for this field?
Respond with ONLY the matched data value from the list provided, and nothing else.
If no value seems appropriate, respond with 'None'."""

    # Prepare multimodal input
    image_part = Part.from_data(data=image_bytes, mime_type="image/png")
    contents = [image_part, prompt]

    # Configure safety settings if needed (optional)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    # Configure generation settings (optional)
    generation_config = GenerationConfig(
        temperature=0.1, # Lower temperature for more deterministic response
        top_p=0.8,
        top_k=5,
        candidate_count=1,
        max_output_tokens=8192,
    )

    try:
        response = model.generate_content(
            contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )
        # Extract the text response
        matched_value_str = response.text.strip()

        # Very basic validation: Check if the response is *exactly* one of the available data items (converted to string)
        # This is crucial because the model might hallucinate or add extra text.
        for item in available_data_list:
            if matched_value_str == str(item):
                 # Find the original item (preserving type, e.g., bool)
                for original_item in available_data_list:
                    if str(original_item) == matched_value_str:
                        print(f"  Gemini matched: {original_item}")
                        return original_item # Return the original item with its type

        print(f"  Gemini response '{matched_value_str}' not found exactly in available data.")
        return None # Indicate no valid match found

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

# --- Phase 1: Generate Plan --- (NEW FUNCTION)
def generate_filling_plan(model, form_image_bytes, data_to_fill):
    """Uses Gemini to create a plan mapping data items to form field descriptions."""
    print("--- Phase 1: Generating Filling Plan with Gemini ---")

    # Prepare prompt
    data_list_str = "\n".join([f"- {repr(item)}" for item in data_to_fill]) # Represent data clearly
    prompt = f"""Analyze the provided form image visually.
Based ONLY on the visual layout and field labels/questions on the form, identify the distinct fillable fields (e.g., 'Full Name text input', '1. Available Nov 1-Apr 24 (Yes checkbox)', '10. Preferred Mon checkbox', '3. Experience text input', etc.).

Now, look at the provided list of data values:
{data_list_str}

Create a plan by matching EACH data value from the list to EXACTLY ONE of the visually identified fields on the form. Ensure the data type seems appropriate for the field type (e.g., boolean for checkbox, string for text input).

Format the output STRICTLY as a JSON object containing two keys:
1.  `"plan"`: An object where keys are descriptive strings uniquely identifying the field on the form (use question number or nearby text for context), and values are the corresponding data values from the list that should go there.
2.  `"unmatched_data"`: A list containing any data values from the provided list that could not be confidently matched to a visual field on the form.

Example of desired JSON output format:
```json
{{
  "plan": {{
    "1. Available Nov 1-Apr 24 (Yes checkbox)": true,
    "3. Experience (text input)": "Server at Cafe Bistro...",
    "10. Preferred Monday (checkbox)": true
  }},
  "unmatched_data": [ "Some leftover data" ]
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
        # response_mime_type="application/json", # Uncomment if supported and desired
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

        try:
            parsed_result = json.loads(json_string)
            # Basic validation of structure
            if isinstance(parsed_result, dict) and "plan" in parsed_result and "unmatched_data" in parsed_result:
                print("Plan generation and JSON parsing successful.")
                # Ensure plan values are drawn from original data list (potential type coercion fix)
                plan_dict = parsed_result.get("plan", {})
                original_data_map = {str(item): item for item in data_to_fill}
                corrected_plan = {}
                for key, value_from_json in plan_dict.items():
                    # Find the original data item to preserve type
                    original_value = original_data_map.get(str(value_from_json))
                    if original_value is not None:
                        corrected_plan[key] = original_value
                    else:
                         # Fallback if somehow value wasn't in original list (shouldn't happen with good prompt)
                         print(f"Warning: Value '{value_from_json}' from plan not found in original data. Using as is.")
                         corrected_plan[key] = value_from_json
                parsed_result["plan"] = corrected_plan
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

# --- Phase 3: Match Plan to Boxes --- (NEW FUNCTION)
def match_plan_to_boxes(model, form_image_bytes, plan, detected_boxes):
    """Uses Gemini to match items in the plan to detected bounding boxes."""
    print("--- Phase 3: Matching Plan to Detected Boxes with Gemini ---")
    final_mapping = {}

    if not detected_boxes:
        print("Warning: No detected boxes provided. Cannot perform matching.")
        return final_mapping

    if not plan:
         print("Warning: Empty plan dictionary provided. Cannot perform matching.")
         return final_mapping

    # Prepare box list string for the prompt (only needs to be done once)
    # Include index for easier reference if needed, though asking for coords is better
    box_list_str = "\n".join([
        f"- Box {i}: Coords: {box['box']}, Label: {box['label']}, Confidence: {box.get('confidence', 'N/A'):.2f}"
        for i, box in enumerate(detected_boxes)
    ])

    image_part = Part.from_data(data=form_image_bytes, mime_type="image/png")

    # Iterate through items in the generated plan dictionary directly
    for field_description, data_value in plan.items():
        print(f"  Matching: '{field_description}' -> '{data_value}'")

        prompt = f"""Context: We are trying to fill a form field based on a plan.
Form Image: Provided
Field Description from Plan: "{field_description}"

List of Detected Bounding Boxes (from an object detection model):
{box_list_str}

Task: Identify the SINGLE best bounding box from the list above that visually corresponds to the "Field Description from Plan" on the provided Form Image. Consider the description, visual location, and box label.

Respond with ONLY the exact coordinates (e.g., [x_min, y_min, x_max, y_max]) of the best matching bounding box from the list. Do not include the box index or label. If no suitable box is found in the list, respond with the exact string 'None'.
"""

        contents = [image_part, prompt] # Include image for visual matching

        # Configure generation
        generation_config = GenerationConfig(
            temperature=0.1, # Low temp for deterministic matching
            candidate_count=1,
            max_output_tokens=8192, # Increase token limit for coordinates
        )
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            # ... other safety settings ...
        }

        try:
            response = model.generate_content(
                contents,
                generation_config=generation_config,
                # safety_settings=safety_settings, # Optional
                stream=False
            )
            response_text = response.text.strip()
            print(f"    Gemini Match Response: {response_text}")

            if response_text.lower() == 'none':
                print(f"    -> Gemini indicated no match found for '{field_description}'")
                continue

            # Attempt to parse coordinates from the response (expecting format like [1.2, 3.4, 5.6, 7.8])
            try:
                # Be robust against minor formatting variations if possible
                response_text = response_text.replace("(", "[").replace(")", "]") # Allow parentheses
                if not (response_text.startswith("[") and response_text.endswith("]")):
                     raise ValueError("Response not in expected list format")

                # Parse numbers, handle potential ints or floats
                coords_list = [float(c.strip()) for c in response_text.strip("[]").split(",")]

                if len(coords_list) == 4:
                    # Find the *original* box dict to preserve exact coords and label info if needed later
                    # This is tricky as float precision might differ slightly.
                    # A safer approach might be to ask Gemini for the *index* instead of coords.
                    # For now, let's use the parsed coords directly. Convert to tuple for dict key.
                    coords_tuple = tuple(coords_list)
                    final_mapping[coords_tuple] = data_value
                    print(f"    -> Matched '{field_description}' to Box Coords: {coords_tuple}")
                else:
                    print(f"    Error: Parsed coordinates list does not have 4 elements: {coords_list}")

            except (ValueError, TypeError) as parse_error:
                print(f"    Error: Could not parse coordinates from Gemini response '{response_text}'. Error: {parse_error}")

        except Exception as e:
            print(f"    Error during Gemini API call for box matching: {e}")
            # Optionally retry?

    print(f"Plan-to-box matching complete. {len(final_mapping)} items successfully mapped.")
    return final_mapping

# --- Phase 4: Draw Final Mapping --- (NEW FUNCTION)
def draw_final_mapping(base_image, final_mapping, font_path, font_size):
    """Draws the data onto the image based on the final box-to-data mapping."""
    print("--- Phase 4: Drawing Final Mapping on Image ---")
    drawn_image = base_image.copy()

    if not final_mapping:
        print("Warning: Final mapping is empty. Nothing to draw.")
        return drawn_image

    items_drawn = 0
    for box_coords_tuple, data_value in final_mapping.items():
        try:
            # Extract coordinates
            x0, y0, x1, y1 = box_coords_tuple
            box_width = x1 - x0
            box_height = y1 - y0

            # Determine field type based on data type primarily
            if isinstance(data_value, bool):
                # Boolean -> Checkbox/Radio
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2
                # Adjust size based on detected box, ensure it's not too large/small
                check_size = max(5, min(box_width, box_height) * 0.7)

                # Draw only if True (common convention, adjust if False needs drawing)
                if data_value:
                    # Using 'fill' for now, could be 'x' or 'checkmark'
                    drawn_image = draw_checkbox_on_image(drawn_image, (center_x, center_y), size=check_size, checked=True, style='fill', color='black')
                    print(f"  -> Drawn checkbox/radio at ~({center_x:.0f}, {center_y:.0f}) for data: {data_value}")
                    items_drawn += 1
                else:
                    # If you need to explicitly mark 'False' (e.g., cross out), add logic here
                    print(f"  -> Skipped drawing checkbox/radio for False value at ~({(x0+x1)/2:.0f}, {(y0+y1)/2:.0f})")
                    # items_drawn += 1 # Count if you consider 'drawing nothing' an action
            elif isinstance(data_value, str):
                # String -> Text Input
                # Position text slightly inside the top-left corner
                text_pos = (x0 + min(5, box_width * 0.1), y0 + min(2, box_height * 0.1))
                drawn_image = draw_text_on_image(drawn_image, data_value, text_pos, font_path=font_path, font_size=font_size, color="black")
                print(f"  -> Drawn text starting at ({text_pos[0]:.0f}, {text_pos[1]:.0f}) for data: '{data_value[:30]}...'")
                items_drawn += 1
            else:
                # Handle other data types if necessary (e.g., numbers)
                print(f"  Warning: Unsupported data type '{type(data_value)}' for drawing at box {box_coords_tuple}. Skipping.")

        except Exception as e:
            print(f"Error drawing data '{data_value}' for box {box_coords_tuple}: {e}")

    print(f"Drawing complete. Attempted to draw {items_drawn} items based on final mapping.")
    return drawn_image

# --- Main Execution --- (Modified)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill a form image using YOLO detection and a Gemini-driven plan.")
    parser.add_argument("--image", required=True, type=str, help="Path to the blank form image file.")
    parser.add_argument("--model", required=True, type=str, help="Path to the trained YOLOv8 model weights (.pt file).")
    parser.add_argument("--output", type=str, default="planned_filled_form.png", help="Path to save the final filled form image.")
    parser.add_argument("--font_path", type=str, default="/Library/Fonts/Arial Unicode.ttf", help="Path to the .ttf font file.")
    parser.add_argument("--font_size", type=int, default=9, help="Font size for drawing text.")
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
    fake_data = [
        True, # Q1 Yes (Nov-Apr)
        True, # Q1 March Yes
        False,# Q1 May No
        True, # Q2 BOTH Yes
        True, # Q2 Transport Yes
        "Server at Cafe Bistro (2 years), Front desk volunteer at community center (1 year)",
        "I enjoy customer interaction and thrive in fast-paced settings. I'm eager to contribute to a positive guest experience at the resort.",
        "Listen actively, apologize sincerely, empathize with their frustration, and offer a concrete solution (e.g., replace item, offer discount). As a customer, I expect to be heard and have the problem resolved efficiently.",
        "Check road conditions beforehand, leave significantly earlier to account for slow travel, ensure my vehicle is equipped for snow, and communicate any potential delays.",
        True, # Q7 Food Handler Yes
        "06/2026", # Q7 Expiration
        False, # Q7 Mixologist No
        "To gain valuable experience in resort hospitality, enhance my customer service skills, and contribute positively to the team environment.",
        "1. Restock condiments/supplies. 2. Wipe down tables/counters. 3. Organize station. 4. Check on guests. 5. Assist colleagues.", # Shortened for brevity
        True, False, False, True, True, True, True, # Q10 Weekdays (Mon-Sun)
        True,  # Q10 Full Time
        True, # Q11 Christmas Yes
        "I am a reliable and enthusiastic worker, keen to learn quickly."
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
    final_mapping = match_plan_to_boxes(match_model, form_image_bytes, filling_plan, detected_bboxes)

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