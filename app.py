
import streamlit as st
import os
import io
import json
import time
import pandas as pd

from PIL import Image, ImageOps
from dotenv import load_dotenv
from google import genai
from google.genai import types


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="GenAI Vision Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. CUSTOM CSS - WHITE AND BLUE DESIGN
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f5f8fc;
}

header[data-testid="stHeader"] {
    background-color: #f5f8fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5edf6;
}

[data-testid="stSidebar"] h1 {
    color: #1764c0;
}

.main-title {
    color: #163b68;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 4px;
}

.subtitle {
    color: #718096;
    font-size: 14px;
    margin-bottom: 25px;
}

.section-title {
    color: #183b63;
    font-size: 20px;
    font-weight: 700;
    margin-top: 12px;
    margin-bottom: 15px;
}

.metric-card {
    background: white;
    border: 1px solid #e3ebf5;
    border-radius: 13px;
    padding: 20px;
    min-height: 110px;
    box-shadow: 0 3px 12px rgba(30, 70, 120, 0.04);
}

.metric-label {
    color: #718096;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 10px;
}

.metric-value {
    color: #174b87;
    font-size: 23px;
    font-weight: 800;
    overflow-wrap: anywhere;
}

.info-card {
    background: white;
    border: 1px solid #e3ebf5;
    border-radius: 13px;
    padding: 20px;
    box-shadow: 0 3px 12px rgba(30, 70, 120, 0.03);
}

.status-ready {
    color: #2563eb;
    background: #eaf2ff;
    padding: 7px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

.status-success {
    color: #15803d;
    background: #e8f8ee;
    padding: 7px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

.status-error {
    color: #b91c1c;
    background: #feecec;
    padding: 7px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

div.stButton > button {
    background-color: #2167c5;
    color: white;
    border: none;
    border-radius: 9px;
    min-height: 45px;
    font-weight: 600;
    transition: 0.2s;
}

div.stButton > button:hover {
    background-color: #174e9b;
    color: white;
    border: none;
}

div.stDownloadButton > button {
    background-color: #eaf2ff;
    color: #1855a0;
    border: 1px solid #c9dcf7;
    border-radius: 9px;
    font-weight: 600;
    min-height: 43px;
}

div.stDownloadButton > button:hover {
    background-color: #dbeaff;
    color: #174e9b;
    border-color: #9fc3f0;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed #8db5e9;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #e3ebf5;
    border-radius: 10px;
}

hr {
    border-color: #e4ebf4;
}

.footer {
    text-align: center;
    color: #8a9bb0;
    font-size: 12px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. APPLICATION SETTINGS
# =====================================================

load_dotenv()

MODEL = "gemini-3.5-flash-lite"

MAX_FILE_SIZE = 15 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1400

SYSTEM_PROMPT = """
You are an expert document information extraction assistant.

Read the uploaded receipt, invoice, bill, or handwritten note.

Extract all visible information.

Return data using exactly this JSON structure:

{
    "vendor": null,
    "date": null,
    "invoice_number": null,
    "customer_name": null,
    "subtotal": null,
    "tax": null,
    "discount": null,
    "total_amount": null,
    "currency": null,
    "line_items": [
        {
            "item": null,
            "quantity": null,
            "unit_price": null,
            "total_price": null
        }
    ],
    "payment_status": "Unknown"
}

RULES:

1. Extract information directly from the image.
2. Never invent missing information.
3. Use null for unavailable fields.
4. Use [] if no line items exist.
5. Extract every visible line item.
6. Preserve original item names.
7. Numerical values must be numbers.
8. Do not change printed totals.
9. Payment status must be based on visible information.
10. Return valid JSON only.
"""


# =====================================================
# 4. SESSION STATE
# =====================================================

if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None

if "processing_time" not in st.session_state:
    st.session_state.processing_time = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "app_status" not in st.session_state:
    st.session_state.app_status = "Ready"

if "vision_result" not in st.session_state:
    st.session_state.vision_result = None


# =====================================================
# 5. API KEY
# =====================================================

def get_api_key():

    key = os.getenv("GEMINI_API_KEY")

    if key:
        return key

    try:
        return st.secrets["GEMINI_API_KEY"]

    except Exception:
        return None


API_KEY = get_api_key()


# =====================================================
# 6. GEMINI CLIENT
# =====================================================

@st.cache_resource
def create_client(api_key):

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120000,
            retry_options=types.HttpRetryOptions(
                attempts=1
            )
        )
    )


# =====================================================
# 7. IMAGE PROCESSING
# =====================================================

@st.cache_data(show_spinner=False)
def process_image(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    image = ImageOps.exif_transpose(image)

    image = image.convert("RGB")

    original_size = image.size

    image.thumbnail(
        (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
        Image.Resampling.LANCZOS
    )

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=85,
        optimize=True
    )

    return (
        output.getvalue(),
        original_size,
        image.size
    )


# =====================================================
# 8. IMAGE INPUT
# =====================================================

def create_image_part(image_bytes):

    return types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/jpeg"
    )


# =====================================================
# 9. VISION RECOGNITION TEST
# =====================================================

def test_vision(image_bytes):

    client = create_client(API_KEY)

    response = client.models.generate_content(

        model=MODEL,

        contents=[
            """
            Examine this image.

            Identify the document type, vendor,
            date, visible items, prices, subtotal,
            tax and total amount.

            Only describe information visible in the image.
            """,

            create_image_part(image_bytes)
        ],

        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text


# =====================================================
# 10. DOCUMENT EXTRACTION
# =====================================================

def extract_document(image_bytes):

    client = create_client(API_KEY)

    start_time = time.perf_counter()

    response = client.models.generate_content(

        model=MODEL,

        contents=[
            SYSTEM_PROMPT,
            create_image_part(image_bytes)
        ],

        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json"
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    data = json.loads(response.text)

    if not isinstance(data, dict):
        raise ValueError("Invalid JSON structure.")

    elapsed = time.perf_counter() - start_time

    return data, elapsed


# =====================================================
# 11. REUSABLE METRIC CARD
# =====================================================

def metric_card(label, value):

    value = value if value is not None else "Not found"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# 12. RESULTS DASHBOARD
# =====================================================

def display_results(data):

    st.markdown(
        '<div class="section-title">Extraction Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        metric_card(
            "VENDOR",
            data.get("vendor")
        )

    with col2:

        total = data.get("total_amount")

        currency = data.get("currency") or ""

        total_value = (
            f"{currency}{total}"
            if total is not None
            else "Not found"
        )

        metric_card(
            "TOTAL AMOUNT",
            total_value
        )

    with col3:

        metric_card(
            "PAYMENT STATUS",
            data.get("payment_status") or "Unknown"
        )

    st.write("")

    # Invoice details

    st.markdown(
        '<div class="section-title">Document Details</div>',
        unsafe_allow_html=True
    )

    details = [
        ("Date", data.get("date")),
        ("Invoice Number", data.get("invoice_number")),
        ("Customer Name", data.get("customer_name")),
        ("Currency", data.get("currency")),
        ("Subtotal", data.get("subtotal")),
        ("Tax", data.get("tax")),
        ("Discount", data.get("discount"))
    ]

    with st.container(border=True):

        col1, col2 = st.columns(2)

        for index, (label, value) in enumerate(details):

            target = col1 if index % 2 == 0 else col2

            with target:

                st.write(
                    f"**{label}**"
                )

                st.write(
                    value if value is not None else "Not found"
                )

                st.divider()

    # Line items

    st.markdown(
        '<div class="section-title">Extracted Line Items</div>',
        unsafe_allow_html=True
    )

    line_items = data.get("line_items", [])

    if isinstance(line_items, list) and line_items:

        df = pd.DataFrame(line_items)

        df = df.rename(
            columns={
                "item": "Item",
                "quantity": "Quantity",
                "unit_price": "Unit Price",
                "total_price": "Total Price"
            }
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"{len(line_items)} line item(s) extracted."
        )

    else:

        st.info("No line items were found.")

    # JSON output

    st.markdown(
        '<div class="section-title">Structured JSON Output</div>',
        unsafe_allow_html=True
    )

    json_output = json.dumps(
        data,
        indent=4,
        ensure_ascii=False
    )

    with st.expander("View complete JSON", expanded=False):

        st.code(
            json_output,
            language="json"
        )

    st.download_button(
        "Download Extracted JSON",
        data=json_output,
        file_name="extracted_document.json",
        mime="application/json",
        use_container_width=True
    )


# =====================================================
# 13. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("# 📄 GenAI Vision")

    st.caption("Document Intelligence Platform")

    st.divider()

    st.markdown("### Navigation")

    st.markdown(
        """
        <div style="
            background:#eaf2ff;
            color:#1855a0;
            padding:12px;
            border-radius:9px;
            font-weight:700;
        ">
        📊 Document Extractor
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown("### Supported Documents")

    st.write("📄 Receipts")
    st.write("🧾 Invoices")
    st.write("✍️ Handwritten Notes")

    st.divider()

    st.markdown("### AI Configuration")

    st.write(f"**Model:** `{MODEL}`")

    st.write("**Technology:** Gemini Vision")

    st.write("**Output:** Structured JSON")

    st.divider()

    if st.session_state.app_status == "Completed":

        st.success("Extraction completed")

    elif st.session_state.app_status == "Failed":

        st.error("Extraction failed")

    elif st.session_state.app_status == "Processing":

        st.info("Processing document")

    else:

        st.info("Ready for upload")


# =====================================================
# 14. MAIN DASHBOARD
# =====================================================

st.markdown(
    '<div class="main-title">Document Intelligence Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Extract meaningful information from documents '
    'using multimodal artificial intelligence.'
    '</div>',
    unsafe_allow_html=True
)

if not API_KEY:

    st.error(
        "Gemini API key not found. Configure "
        "GEMINI_API_KEY in your .env file."
    )

    st.stop()


# =====================================================
# 15. UPLOAD AREA
# =====================================================

st.markdown(
    '<div class="section-title">Upload Your Document</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Select a receipt, invoice or handwritten note",
    type=["jpg", "jpeg", "png", "webp"],
    help="Maximum supported file size: 15 MB"
)


if uploaded_file is not None:

    if uploaded_file.size > MAX_FILE_SIZE:

        st.error(
            "File exceeds the 15 MB size limit."
        )

        st.stop()

    original_bytes = uploaded_file.getvalue()

    # Process uploaded image

    try:

        with st.spinner("Preparing document image..."):

            (
                processed_bytes,
                original_size,
                processed_size
            ) = process_image(original_bytes)

    except Exception as error:

        st.error(f"Image processing failed: {error}")

        st.stop()

    # Reset results when a different document is uploaded

    if st.session_state.document_name != uploaded_file.name:

        st.session_state.document_name = uploaded_file.name
        st.session_state.extracted_data = None
        st.session_state.processing_time = None
        st.session_state.vision_result = None
        st.session_state.app_status = "Ready"

    # Preview section

    col1, col2 = st.columns([1.1, 1])

    with col1:

        with st.container(border=True):

            st.markdown("#### Document Preview")

            st.image(
                processed_bytes,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.markdown("#### File Information")

            st.write(
                f"**File name:** {uploaded_file.name}"
            )

            st.write(
                f"**Original dimensions:** "
                f"{original_size[0]} × {original_size[1]}"
            )

            st.write(
                f"**Optimized dimensions:** "
                f"{processed_size[0]} × {processed_size[1]}"
            )

            st.write(
                f"**Original size:** "
                f"{uploaded_file.size / 1024:.1f} KB"
            )

            st.write(
                f"**Processed size:** "
                f"{len(processed_bytes) / 1024:.1f} KB"
            )

            st.divider()

            st.markdown("**Processing status**")

            if st.session_state.app_status == "Completed":

                st.markdown(
                    '<span class="status-success">'
                    '✓ Completed'
                    '</span>',
                    unsafe_allow_html=True
                )

            elif st.session_state.app_status == "Failed":

                st.markdown(
                    '<span class="status-error">'
                    '✕ Failed'
                    '</span>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<span class="status-ready">'
                    '● Ready'
                    '</span>',
                    unsafe_allow_html=True
                )

    st.write("")

    # Action buttons

    col1, col2 = st.columns(2)

    with col1:

        test_button = st.button(
            "Test Vision Recognition",
            use_container_width=True
        )

    with col2:

        extract_button = st.button(
            "Extract Document Data",
            type="primary",
            use_container_width=True
        )

    # Vision recognition

    if test_button:

        try:

            with st.spinner(
                "Gemini is recognizing your document..."
            ):

                start_time = time.perf_counter()

                vision_result = test_vision(
                    processed_bytes
                )

                elapsed = time.perf_counter() - start_time

            st.session_state.vision_result = vision_result

            st.success(
                f"Vision test completed in {elapsed:.1f} seconds."
            )

        except Exception as error:

            st.error(
                f"Vision recognition failed: {error}"
            )

    if st.session_state.vision_result:

        with st.expander(
            "Gemini Vision Recognition Result",
            expanded=True
        ):

            st.write(
                st.session_state.vision_result
            )

    # Document extraction

    if extract_button:

        st.session_state.app_status = "Processing"

        try:

            with st.spinner(
                "Gemini is analyzing your document and extracting data..."
            ):

                result, elapsed = extract_document(
                    processed_bytes
                )

            st.session_state.extracted_data = result

            st.session_state.processing_time = elapsed

            st.session_state.app_status = "Completed"

            st.success(
                f"Document successfully extracted in "
                f"{elapsed:.1f} seconds."
            )

        except Exception as error:

            st.session_state.extracted_data = None

            st.session_state.app_status = "Failed"

            st.error(
                f"Document extraction failed: {error}"
            )

            message = str(error).lower()

            if "timeout" in message or "504" in message:

                st.warning(
                    "Gemini took too long to respond. "
                    "Please try again later."
                )

            elif "503" in message:

                st.warning(
                    "Gemini is experiencing high demand. "
                    "Please wait and try again."
                )

            elif "429" in message:

                st.warning(
                    "API request limit reached. "
                    "Wait before trying again."
                )

            with st.expander("Technical Error Details"):

                st.exception(error)

    # Display extraction results

    if st.session_state.extracted_data is not None:

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            'Extraction Results'
            '</div>',
            unsafe_allow_html=True
        )

        if st.session_state.processing_time is not None:

            st.caption(
                f"Processing time: "
                f"{st.session_state.processing_time:.1f} seconds"
            )

        display_results(
            st.session_state.extracted_data
        )

else:

    st.markdown(
        """
        <div style="
            background:white;
            border:1px dashed #9bbde8;
            border-radius:15px;
            padding:45px 20px;
            text-align:center;
            margin-top:15px;
        ">
            <div style="font-size:45px;">📄</div>
            <h3 style="color:#174b87;">
                Ready to Extract Document Information
            </h3>
            <p style="color:#718096;">
                Upload a document to start AI-powered extraction.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# 16. FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        GenAI Vision Extractor | Week 7 - Multimodal AI & Vision
        <br>
        Powered by Google Gemini
    </div>
    """,
    unsafe_allow_html=True
)
