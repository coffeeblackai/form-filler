# Automated PDF Form Filler

This project automates the process of filling PDF form *images* using a combination of AI techniques:

1.  **Object Detection (YOLOv8):** Detects the locations of form fields (text inputs, checkboxes, etc.) on the input image.
2.  **Large Language Model (Google Gemini):**
    *   Analyzes the form visually and creates a logical plan to match provided data values to the identified fields based on context and labels.
    *   Matches the items in the generated plan to the specific bounding boxes detected by YOLO.
3.  **Image Manipulation (Pillow):** Draws the text and checkbox/radio button selections onto the final output image.

## Features

*   Detects various form fields (text inputs, checkboxes, radio buttons) using a trained YOLOv8 model.
*   Uses Gemini's multimodal capabilities to understand form structure and create a robust filling plan.
*   Matches the plan to detected bounding boxes, providing resilience against inaccurate or missing detections.
*   Generates an output image with the data visually filled in.
*   Includes scripts for model inference, training, and potentially data preparation.

## Workflow (`fill_form.py`)

The main script (`fill_form.py`) orchestrates the following workflow:

1.  **Load Input:** Reads the blank form image and the data intended for filling.
2.  **Phase 1: Generate Plan:** Sends the form image and data list to Gemini (`PLAN_MODEL_ID`). Gemini returns a JSON plan mapping descriptive field identifiers (e.g., "3. Experience text input") to the corresponding data values.
3.  **Phase 2: Detect Boxes:** Runs YOLOv8 inference (`scripts/inference.py`) on the form image to detect bounding boxes for potential fields.
4.  **Phase 3: Match Plan to Boxes:** For each item in the Gemini-generated plan, sends the form image, the field description, and the list of detected boxes to Gemini (`MATCH_MODEL_ID`). Gemini identifies the specific bounding box coordinates that best match the field description.
5.  **Phase 4: Draw Output:** Iterates through the final mapping (box coordinates -> data value) generated in Phase 3. Uses Pillow to draw the text or checkbox/radio button fills onto a copy of the original image at the matched coordinates.
6.  **Save Output:** Saves the final image with the drawn data.

## Setup

1.  **Clone Repository:**
    ```bash
    git clone <your-repository-url>
    cd pdf-form-filler
    ```

2.  **Python Environment:** Ensure you have Python 3.8+ installed. Consider using a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You may need to create a `requirements.txt` file. Based on the scripts, it should include at least:*
    ```
    # requirements.txt
    ultralytics
    google-cloud-aiplatform
    Pillow
    opencv-python
    # Add other dependencies if used (e.g., Faker for fill_form_test.py if that exists)
    ```
    *)*

4.  **Google Cloud Authentication:** Authenticate your environment to use Google Cloud services (Vertex AI Gemini).
    ```bash
    gcloud auth application-default login
    ```

5.  **Google Cloud Project:** Set the `GOOGLE_CLOUD_PROJECT` environment variable or hardcode your Project ID in `fill_form.py`.
    ```bash
    export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
    ```

6.  **YOLOv8 Model:** Obtain a trained YOLOv8 model file (`.pt`) capable of detecting the form field classes (`text_input`, `checkbox`, `radio_button`, etc.). You might need to train one using `scripts/train.py`. Place the `.pt` file where the scripts can access it (e.g., in the root directory or a dedicated `models/` folder).

7.  **Font:** Ensure the font path specified in `fill_form.py` (default: `/Library/Fonts/Arial Unicode.ttf`) is valid for your system, or update the path/default argument.

## Usage

### Filling a Form (`fill_form.py`)

This is the main script to run the end-to-end form filling process.

```bash
python fill_form.py \
    --image <path/to/form_image.png> \
    --model <path/to/your_model.pt> \
    --output <path/to/save/filled_form.png> \
    --conf <confidence_threshold> \
    --font_path <path/to/font.ttf> \
    --font_size <size>
```

**Arguments:**

*   `--image` (Required): Path to the blank form image.
*   `--model` (Required): Path to the trained YOLOv8 model weights (`.pt`).
*   `--output` (Optional): Path to save the final filled image. Defaults to `planned_filled_form.png`.
*   `--conf` (Optional): Confidence threshold for YOLO detections (0.0 to 1.0). Defaults to `0.3`. Lower values detect more boxes but may include false positives.
*   `--font_path` (Optional): Path to the `.ttf` font file for drawing text. Defaults to `/Library/Fonts/Arial Unicode.ttf`.
*   `--font_size` (Optional): Font size for drawing text. Defaults to `9`.
*   `--annotated_detect_output` (Optional): Path to save an intermediate image showing the raw YOLO detections. Defaults to `detected_boxes.png`.

**Note:** The data to be filled is currently hardcoded within `fill_form.py` in the `fake_data` list. For real-world use, this should be loaded dynamically (e.g., from a JSON file, database, or user input).

## Scripts

### `fill_form.py`

*   **Purpose:** Main orchestrator for the form filling process using the multi-stage AI approach.
*   **Usage:** See "Usage" section above.

### `scripts/inference.py`

*   **Purpose:** Performs object detection using a trained YOLOv8 model on a given image. Saves an image with bounding boxes drawn and can be used as a library function (as done by `fill_form.py`) to return the detected box data.
*   **Standalone Usage:**
    ```bash
    python scripts/inference.py \
        --image <path/to/input_image.png> \
        --model <path/to/your_model.pt> \
        --output <path/to/save/detection_output.png> \
        --conf <confidence_threshold>
    ```

### `scripts/train.py`

*   **Purpose:** Trains a YOLOv8 detection model on a custom dataset for form fields. Requires a dataset configured in YOLO format (usually with a `.yaml` file specifying paths and class names).
*   **Example Usage:**
    ```bash
    python scripts/train.py \
        --data <path/to/dataset.yaml> \
        --epochs <number_of_epochs> \
        --imgsz <image_size> \
        --name <run_name>
        # Add other YOLO training arguments as needed (e.g., --batch, --weights)
    ```

### `scripts/download_datasets.py`

*   **Purpose:** Helper script likely used to download datasets required for training the YOLO model. The specific implementation details (sources, arguments) depend on how it was written.
*   **Hypothetical Usage:**
    ```bash
    python scripts/download_datasets.py --dataset_name <some_dataset> --output_dir ./datasets
    ```

### `scripts/extract_pdf_fields.py`

*   **Purpose:** Likely extracts information *from* PDF documents, possibly field definitions, locations, or values. This might be used for creating training data annotations, evaluating filling accuracy, or other preprocessing steps. The specific functionality depends on its implementation.
*   **Hypothetical Usage:**
    ```bash
    python scripts/extract_pdf_fields.py --pdf_input <path/to/document.pdf> --output_json <path/to/save/extracted_data.json>
    ```

## Configuration

Key configuration parameters can be found and modified near the top of `fill_form.py`:

*   `PROJECT_ID`: Your Google Cloud Project ID.
*   `LOCATION`: Google Cloud region for Vertex AI services (e.g., `us-central1`, `us-west4`).
*   `PLAN_MODEL_ID`: The Gemini model used for generating the filling plan (Phase 1).
*   `MATCH_MODEL_ID`: The Gemini model used for matching the plan to bounding boxes (Phase 3).

## Future Improvements

*   Load input data dynamically instead of using the hardcoded `fake_data`.
*   Implement more robust error handling and retries for API calls.
*   Explore alternative strategies for matching plan items to boxes (e.g., using box indices, geometric heuristics).
*   Add support for different field types (dropdowns, signatures if detected).
*   Develop a user interface.
*   Add evaluation metrics to compare filled forms against ground truth. 