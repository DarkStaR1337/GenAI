import concurrent.futures
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from pydantic import BaseModel

# ============================================================================
# GLOBAL SETTINGS AND INITIALIZATION
# ============================================================================
ROOT_DIR = r"PATH_TO_PO_FILES"
DB_FILE = "user.db"

# Logging Configuration
LOG_FILE = os.path.join(ROOT_DIR, "po_runtime.log")
os.makedirs(ROOT_DIR, exist_ok=True)
logger = logging.getLogger("po_runtime")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

# Azure Resources
document_client = DocumentAnalysisClient(
    endpoint="YOUR_AZURE_ENDPOINT", 
    credential=AzureKeyCredential("YOUR_AZURE_KEY")
)

# Azure OpenAI for translation
azure_openai_client = AzureOpenAI(
    azure_endpoint="YOUR_AZURE_OPENAI_ENDPOINT",
    api_key="YOUR_AZURE_OPENAI_KEY",
    api_version="2024-02-15-preview"
)

MAX_WORKERS = 3
TRANSLATION_CACHE = {}  # Cache translations to avoid redundant API calls
translation_lock = threading.Lock()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class POInfo(BaseModel):
    filename: str
    po_number: Optional[str]
    extracted_data: Dict[str, Any]
    page_count: int
    page_range: str
    original_languages: List[str]
    processed_at: float
    file_hash: str
    sequence_number: int

class ProcessingStatus(BaseModel):
    total_files: int
    processed_files: int
    pending_files: int
    recently_processed: List[str]
    error_files: List[str]
    is_running: bool

class SplitResult(BaseModel):
    original_filename: str
    total_pos_found: int
    po_files: List[Dict[str, Any]]
    processing_time: float

# ============================================================================
# AI TRANSLATION LAYER
# ============================================================================

def detect_language(text: str) -> str:
    """
    Detect the language of the text using Azure OpenAI GPT-4.
    Returns language code (e.g., 'es', 'de', 'zh', 'ja', 'en')
    """
    try:
        response = azure_openai_client.chat.completions.create(
            model="gpt-4o",  # Use your GPT-4o deployment name
            messages=[
                {
                    "role": "system",
                    "content": "You are a language detection expert. Respond with ONLY the ISO 639-1 language code (e.g., 'en', 'es', 'de', 'zh', 'ja', 'fr', 'it', 'pt', 'ru', 'ar', 'hi'). Nothing else."
                },
                {
                    "role": "user",
                    "content": f"Detect the language of this text:\n\n{text[:1000]}"
                }
            ],
            temperature=0,
            max_tokens=10
        )
        
        language_code = response.choices[0].message.content.strip().lower()
        logger.info(f"Detected language: {language_code}")
        return language_code
        
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return "unknown"


def translate_text_with_ai(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """
    Translate text to English using Azure OpenAI GPT-4o while preserving:
    - Document structure
    - Numbers, codes, and identifiers
    - Field names and labels
    - Table formatting
    """
    # Check cache first
    cache_key = hashlib.md5(f"{text}{source_lang}{target_lang}".encode()).hexdigest()
    
    with translation_lock:
        if cache_key in TRANSLATION_CACHE:
            logger.info(f"Using cached translation for text hash: {cache_key[:8]}...")
            return TRANSLATION_CACHE[cache_key]
    
    try:
        response = azure_openai_client.chat.completions.create(
            model="gpt-4o",  # Use your GPT-4o deployment name
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional document translator specializing in business documents (Purchase Orders, Invoices, etc.).

CRITICAL RULES:
1. Translate ALL text to English
2. PRESERVE exactly as-is:
   - All numbers (dates, amounts, quantities, codes)
   - All alphanumeric identifiers (PO numbers, item codes, reference numbers)
   - All special characters and formatting
   - Table structures and layouts
3. Translate field labels/headers but keep their position
4. Maintain the exact document structure
5. Do not add explanations or notes
6. Return ONLY the translated text"""
                },
                {
                    "role": "user",
                    "content": f"Translate this document text from {source_lang} to English:\n\n{text}"
                }
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        translated_text = response.choices[0].message.content.strip()
        
        # Cache the translation
        with translation_lock:
            TRANSLATION_CACHE[cache_key] = translated_text
        
        logger.info(f"Translated text from {source_lang} to English ({len(text)} -> {len(translated_text)} chars)")
        return translated_text
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text  # Return original text if translation fails


def translate_pdf_page(page_bytes: bytes) -> Tuple[bytes, str]:
    """
    Translate a PDF page by:
    1. Extracting text using OCR (Azure DI)
    2. Detecting language
    3. Translating to English if needed
    4. Creating overlay with translated text
    
    Returns: (modified_pdf_bytes, detected_language)
    """
    try:
        # Extract text and layout using Azure DI
        poller = document_client.begin_analyze_document(
            "prebuilt-layout",
            page_bytes
        )
        result = poller.result()
        
        if not result.pages or len(result.pages) == 0:
            logger.warning("No pages found in document")
            return page_bytes, "unknown"
        
        page = result.pages[0]
        
        # Extract all text from the page
        page_text = ""
        text_elements = []
        
        for line in page.lines if hasattr(page, 'lines') else []:
            page_text += line.content + "\n"
            text_elements.append({
                'content': line.content,
                'polygon': line.polygon,
                'bbox': line.bounding_box if hasattr(line, 'bounding_box') else None
            })
        
        if not page_text.strip():
            logger.info("No text found on page")
            return page_bytes, "unknown"
        
        # Detect language
        detected_lang = detect_language(page_text)
        
        # If already in English, return as-is
        if detected_lang in ['en', 'unknown']:
            logger.info("Page is already in English or language unknown, skipping translation")
            return page_bytes, detected_lang
        
        # Translate the text
        translated_text = translate_text_with_ai(page_text, detected_lang, "en")
        
        # Create a new PDF with translated text overlay
        # For simplicity, we'll create a text layer PDF
        # In production, you might want to use a more sophisticated approach
        translated_pdf_bytes = create_translated_pdf_overlay(page_bytes, text_elements, page_text, translated_text)
        
        logger.info(f"Successfully translated page from {detected_lang} to English")
        return translated_pdf_bytes, detected_lang
        
    except Exception as e:
        logger.error(f"Error translating PDF page: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return page_bytes, "error"


def create_translated_pdf_overlay(original_pdf_bytes: bytes, text_elements: List[Dict], 
                                  original_text: str, translated_text: str) -> bytes:
    """
    Create a translated PDF by overlaying translated text on the original PDF.
    This is a simplified version - in production you'd want more sophisticated layout preservation.
    """
    try:
        # Open original PDF
        doc = fitz.open(stream=original_pdf_bytes, filetype="pdf")
        page = doc[0]
        
        # For now, we'll add translated text as a text layer
        # A more sophisticated approach would map individual text elements to their translations
        
        # Add invisible text layer with translation for searchability and DI
        text_rect = page.rect
        page.insert_textbox(
            text_rect,
            translated_text,
            fontsize=1,  # Very small, nearly invisible
            color=(1, 1, 1),  # White text
            overlay=False  # Place underneath
        )
        
        # Save modified PDF
        translated_bytes = doc.write()
        doc.close()
        
        return bytes(translated_bytes)
        
    except Exception as e:
        logger.error(f"Error creating translated PDF overlay: {e}")
        return original_pdf_bytes


def translate_pdf_pages_batch(page_bytes_list: List[bytes]) -> Tuple[List[bytes], Dict[int, str]]:
    """
    Translate multiple PDF pages in batch, maintaining order.
    
    Returns: 
        - List of translated page bytes (in same order)
        - Dict mapping page index to detected language
    """
    logger.info(f"Starting batch translation of {len(page_bytes_list)} pages")
    
    translated_pages = []
    language_map = {}
    
    # Process pages in order (can be parallelized if needed)
    for idx, page_bytes in enumerate(page_bytes_list):
        try:
            translated_bytes, detected_lang = translate_pdf_page(page_bytes)
            translated_pages.append(translated_bytes)
            language_map[idx] = detected_lang
            logger.info(f"Page {idx + 1}/{len(page_bytes_list)} processed (language: {detected_lang})")
        except Exception as e:
            logger.error(f"Error translating page {idx}: {e}")
            # Use original page if translation fails
            translated_pages.append(page_bytes)
            language_map[idx] = "error"
    
    logger.info(f"Batch translation complete. Languages detected: {set(language_map.values())}")
    return translated_pages, language_map

def split_pdf_pages(pdf_bytes: bytes) -> Optional[List[bytes]]:
    """
    Split a multi-page PDF into individual single-page PDFs.
    Returns a list of byte arrays, each representing one page.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        logger.info(f"Opening PDF: size={len(pdf_bytes)} bytes, pages={doc.page_count}")
    except Exception as e:
        logger.error(f"split_pdf_pages: Couldn't open PDF: {e}")
        return None

    page_byte_list = []
    for page_num in range(doc.page_count):
        single_pdf = fitz.open()
        try:
            single_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
            if single_pdf.page_count != 1:
                logger.error(f"Failed to extract page {page_num} (got {single_pdf.page_count} pages)")
                continue
            buf = single_pdf.write()
            page_byte_list.append(bytes(buf))
        except Exception as e:
            logger.error(f"split_pdf_pages: Couldn't extract page {page_num}: {e}")
        finally:
            single_pdf.close()
    
    doc.close()
    logger.info(f"Successfully split PDF into {len(page_byte_list)} pages")
    return page_byte_list


def is_likely_po_start_page(page_bytes: bytes) -> bool:
    """
    Determine if a page is likely the start of a new PO document.
    Uses Azure DI to detect document structure. Works on translated (English) pages.
    """
    try:
        poller = document_client.begin_analyze_document(
            "prebuilt-layout",
            page_bytes
        )
        result = poller.result()
        
        if result.pages:
            first_page = result.pages[0]
            # Get text from top 30% of page (typical header area)
            top_lines = []
            page_height = first_page.height if hasattr(first_page, 'height') else 1000
            threshold = page_height * 0.3
            
            for line in first_page.lines if hasattr(first_page, 'lines') else []:
                if hasattr(line, 'polygon') and line.polygon and len(line.polygon) >= 2:
                    y_coord = line.polygon[0].y if hasattr(line.polygon[0], 'y') else 0
                    if y_coord < threshold:
                        top_lines.append(line.content)
            
            header_text = " ".join(top_lines).upper()
            
            # Since pages are translated to English, we can use consistent English keywords
            po_indicators = [
                'PURCHASE ORDER', 'PO ', 'P.O.', 'ORDER NUMBER', 'ORDER NO',
                'BUYER', 'VENDOR', 'SHIP TO', 'BILL TO', 'SUPPLIER'
            ]
            
            return any(indicator in header_text for indicator in po_indicators)
            
    except Exception as e:
        logger.warning(f"Error in PO start page detection: {e}")
    
    return False


def extract_po_number(page_bytes: bytes) -> Optional[str]:
    """
    Extract PO number from a single page using Azure Document Intelligence.
    Language-agnostic approach using custom model or field detection.
    """
    try:
        # Use custom PO model if available, otherwise use prebuilt-document
        poller = document_client.begin_analyze_document(
            "prebuilt-document",  # Replace with your custom PO model name
            page_bytes
        )
        result = poller.result()
        
        # Strategy 1: Look for PO-specific fields from custom model
        for doc in result.documents:
            for key in ["PONumber", "PurchaseOrderNumber", "PO_Number", "OrderNumber"]:
                val = doc.fields.get(key)
                if val and getattr(val, "value", None):
                    po_num = str(val.value).strip()
                    logger.info(f"Found PO number via DI field: {po_num}")
                    return po_num
        
        # Strategy 2: Check key-value pairs for PO-related patterns
        # This works across languages as DI identifies key-value structure
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.value:
                key_content = kv_pair.key.content
                value_content = kv_pair.value.content.strip()
                
                # Check if key contains any numeric/alphanumeric pattern that looks like a label
                # and value looks like a PO number (alphanumeric, typically 5-20 chars)
                if value_content and len(value_content) >= 3 and len(value_content) <= 30:
                    # Check if value matches common PO number patterns
                    if re.match(r'^[A-Z0-9][A-Z0-9\-\/\_\.]*[A-Z0-9]


def detect_po_boundaries(page_bytes_list: List[bytes]) -> List[Tuple[int, int]]:
    """
    Detect PO boundaries by identifying start pages of new POs.
    Returns list of tuples (start_page_index, end_page_index) for each PO.
    
    This ensures:
    1. Pages are processed in order (0 to n)
    2. Each PO is contiguous from start to end
    3. No page reordering or shuffling
    """
    boundaries = []
    current_start = 0
    
    logger.info(f"Detecting PO boundaries for {len(page_bytes_list)} pages")
    
    for i in range(len(page_bytes_list)):
        # Check if this page marks the start of a new PO
        if i > 0 and is_likely_po_start_page(page_bytes_list[i]):
            # Found a new PO start, so previous PO ends at i-1
            boundaries.append((current_start, i - 1))
            logger.info(f"PO boundary detected: pages {current_start} to {i-1}")
            current_start = i
    
    # Add the last PO (from current_start to end)
    if current_start < len(page_bytes_list):
        boundaries.append((current_start, len(page_bytes_list) - 1))
        logger.info(f"Final PO boundary: pages {current_start} to {len(page_bytes_list)-1}")
    
    logger.info(f"Total PO boundaries detected: {len(boundaries)}")
    return boundaries


def group_pages_by_po(page_bytes_list: List[bytes]) -> List[Dict[str, Any]]:
    """
    Group pages by PO, maintaining strict page order.
    Uses boundary detection to identify where each PO starts and ends.
    
    Returns: List of dicts with keys 'po_number', 'pages', 'page_range'
    """
    if not page_bytes_list:
        return []
    
    groups = []
    
    # Detect PO boundaries (where each PO starts and ends)
    boundaries = detect_po_boundaries(page_bytes_list)
    
    # Process each detected PO sequentially
    for idx, (start_idx, end_idx) in enumerate(boundaries):
        # Extract pages for this PO (maintaining order)
        po_pages = page_bytes_list[start_idx:end_idx + 1]
        
        # Extract PO number from the first page of this group
        po_number = extract_po_number(po_pages[0])
        
        # If no PO number found on first page, try second page
        if not po_number and len(po_pages) > 1:
            po_number = extract_po_number(po_pages[1])
        
        groups.append({
            'po_number': po_number,
            'pages': po_pages,
            'page_count': len(po_pages),
            'page_range': f"{start_idx + 1}-{end_idx + 1}",  # Human-readable (1-indexed)
            'original_page_indices': list(range(start_idx, end_idx + 1))
        })
        
        logger.info(f"PO Group {idx + 1}: pages {start_idx + 1}-{end_idx + 1}, "
                   f"PO number: {po_number if po_number else 'Not found'}, "
                   f"page count: {len(po_pages)}")
    
    return groups


def assemble_pdf_from_pages(page_bytes_group: List[bytes]) -> bytes:
    """
    Assemble multiple single-page PDFs into one multi-page PDF.
    """
    pdf = fitz.open()
    
    for idx, page_bytes in enumerate(page_bytes_group):
        try:
            single_doc = fitz.open(stream=page_bytes, filetype='pdf')
            pdf.insert_pdf(single_doc, from_page=0, to_page=0)
            single_doc.close()
        except Exception as e:
            logger.error(f"Error assembling page {idx}: {e}")
    
    out_bytes = pdf.write()
    pdf.close()
    return bytes(out_bytes)


def extract_pos_from_pdf(pdf_bytes: bytes, original_filename: str) -> List[Tuple[bytes, str, Optional[str], str, Dict[int, str]]]:
    """
    Main function to extract individual POs from a multi-PO PDF.
    Now includes AI translation layer before processing.
    
    Returns: List of tuples (pdf_bytes, filename, po_number, page_range, language_map)
    """
    logger.info(f"Starting PO extraction for: {original_filename}")
    logger.info(f"PDF size: {len(pdf_bytes)} bytes")
    
    # Step 1: Split into individual pages (maintaining order)
    page_bytes_list = split_pdf_pages(pdf_bytes)
    if not page_bytes_list:
        logger.error("Failed to split PDF into pages")
        return []
    
    logger.info(f"Split PDF into {len(page_bytes_list)} pages")
    
    # Step 2: TRANSLATION LAYER - Translate all pages to English
    logger.info("=" * 80)
    logger.info("STARTING AI TRANSLATION LAYER")
    logger.info("=" * 80)
    translated_pages, language_map = translate_pdf_pages_batch(page_bytes_list)
    logger.info(f"Translation complete. Detected languages: {set(language_map.values())}")
    logger.info("=" * 80)
    
    # Step 3: Group pages by PO (using translated pages for consistent extraction)
    groups = group_pages_by_po(translated_pages)
    if not groups:
        logger.error("No PO groups found")
        return []
    
    logger.info(f"Identified {len(groups)} PO groups")
    
    # Step 4: Assemble each group into a separate PDF (in order)
    po_files = []
    base_name = os.path.splitext(original_filename)[0]
    
    for idx, group in enumerate(groups, 1):
        if not group['pages']:
            logger.warning(f"Group {idx} has no pages, skipping")
            continue
        
        try:
            # Assemble PDF from translated pages (pages are already in correct order)
            po_pdf_bytes = assemble_pdf_from_pages(group['pages'])
            
            # Generate filename
            if group['po_number']:
                # Sanitize PO number for filename
                safe_po_num = re.sub(r'[^\w\-]', '_', group['po_number'])
                po_filename = f"{base_name}_PO{idx}_{safe_po_num}.pdf"
            else:
                po_filename = f"{base_name}_PO{idx}_Unknown.pdf"
            
            page_range = group['page_range']
            
            # Get language info for this group
            group_languages = {language_map.get(i, 'unknown') for i in group['original_page_indices']}
            
            po_files.append((po_pdf_bytes, po_filename, group['po_number'], page_range, language_map))
            
            logger.info(f"✓ Created PO {idx}: {po_filename}")
            logger.info(f"  - Pages: {page_range} ({group['page_count']} pages)")
            logger.info(f"  - PO Number: {group['po_number'] if group['po_number'] else 'Not detected'}")
            logger.info(f"  - Languages: {', '.join(group_languages)}")
            logger.info(f"  - Size: {len(po_pdf_bytes)} bytes")
            
        except Exception as e:
            logger.error(f"Error creating PO file for group {idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"Successfully extracted {len(po_files)} PO files from {original_filename}")
    logger.info("=" * 80)
    
    return po_files


# ============================================================================
# FILE PROCESSING AND EXTRACTION
# ============================================================================

def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


def get_user_directories(user_directory: str) -> Dict[str, str]:
    """Get or create user-specific directories."""
    sanitized = re.sub(r'\W+', '_', user_directory)
    base = os.path.join(ROOT_DIR, "user_data", sanitized)
    upload = os.path.join(base, "PO_data")
    extracted = os.path.join(base, "Extracted_data")
    seq_file = os.path.join(base, "sequence_mapping.json")
    
    for d in [upload, extracted]:
        os.makedirs(d, exist_ok=True)
    
    return {
        "base": base,
        "upload": upload,
        "extracted": extracted,
        "sequence_file": seq_file
    }


def save_po_file(file_content: bytes, filename: str, dirs: Dict[str, str]) -> str:
    """Save PO file to user's upload directory."""
    file_path = os.path.join(dirs["upload"], filename)
    with open(file_path, 'wb') as f:
        f.write(file_content)
    logger.info(f"Saved PO file: {file_path}")
    return file_path


def extract_po_data(file_content: bytes, filename: str, dirs: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract structured data from a PO using Azure Document Intelligence.
    """
    extract_file_path = os.path.join(dirs["extracted"], f"{os.path.splitext(filename)[0]}_extracted.json")
    
    # Check if already extracted
    if os.path.exists(extract_file_path):
        try:
            with open(extract_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading existing extracted file: {e}")
    
    try:
        # Analyze document with Azure DI
        poller = document_client.begin_analyze_document(
            "prebuilt-document",  # Or use custom PO model
            file_content
        )
        result = poller.result()
        
        # Extract fields
        extracted_fields = {}
        for doc in result.documents:
            for field_name, field_value in doc.fields.items():
                if hasattr(field_value, 'value') and field_value.value is not None:
                    extracted_fields[field_name] = field_value.value
        
        # Extract tables
        extracted_tables = []
        for table_idx, table in enumerate(result.tables):
            table_data = []
            for row_idx in range(table.row_count):
                row_data = []
                for col_idx in range(table.column_count):
                    cell_text = ""
                    for cell in table.cells:
                        if cell.row_index == row_idx and cell.column_index == col_idx:
                            cell_text = cell.content
                            break
                    row_data.append(cell_text)
                table_data.append(row_data)
            
            extracted_tables.append({
                "table_index": table_idx,
                "data": table_data,
                "row_count": table.row_count,
                "column_count": table.column_count
            })
        
        # Extract key-value pairs
        key_value_pairs = []
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.value:
                key_value_pairs.append({
                    "key": kv_pair.key.content,
                    "value": kv_pair.value.content
                })
        
        extracted_data = {
            "fields": extracted_fields,
            "tables": extracted_tables,
            "key_value_pairs": key_value_pairs,
            "document_type": result.documents[0].doc_type if result.documents else "Unknown"
        }
        
        # Save extracted data
        with open(extract_file_path, 'w') as f:
            json.dump(extracted_data, f, default=str, indent=2)
        
        logger.info(f"Extracted data saved: {extract_file_path}")
        return extracted_data
        
    except Exception as e:
        logger.error(f"Error extracting PO data: {e}")
        return {"error": str(e)}


def process_single_po(po_bytes: bytes, po_filename: str, po_number: Optional[str], 
                      sequence_num: int, page_range: str, original_languages: List[str],
                      dirs: Dict[str, str]) -> POInfo:
    """Process a single PO file and return POInfo."""
    
    logger.info(f"Processing PO {sequence_num}: {po_filename}")
    
    # Save file
    save_po_file(po_bytes, po_filename, dirs)
    
    # Calculate hash
    file_hash = calculate_file_hash(po_bytes)
    
    # Extract data (from translated PDF)
    extracted_data = extract_po_data(po_bytes, po_filename, dirs)
    
    # Count pages
    try:
        doc = fitz.open(stream=po_bytes, filetype="pdf")
        page_count = doc.page_count
        doc.close()
    except:
        page_count = 0
    
    logger.info(f"✓ Processed: {po_filename} ({page_count} pages, hash: {file_hash[:8]}...)")
    
    return POInfo(
        filename=po_filename,
        po_number=po_number,
        extracted_data=extracted_data,
        page_count=page_count,
        page_range=page_range,
        original_languages=original_languages,
        processed_at=time.time(),
        file_hash=file_hash,
        sequence_number=sequence_num
    )


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Multi-PO PDF Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/split-and-process-pos")
async def split_and_process_pos(
    file: UploadFile = File(...),
    user_id: str = "default_user"
):
    """
    Upload a multi-PO PDF and split it into individual PO files.
    Each PO is processed and data is extracted.
    """
    start_time = time.time()
    
    try:
        # Read uploaded file
        pdf_bytes = await file.read()
        original_filename = file.filename
        
        logger.info(f"Received file: {original_filename}, size: {len(pdf_bytes)} bytes")
        
        # Get user directories
        dirs = get_user_directories(user_id)
        
        # Extract POs from PDF
        po_files = extract_pos_from_pdf(pdf_bytes, original_filename)
        
        if not po_files:
            raise HTTPException(status_code=400, detail="No POs found in PDF")
        
        # Process each PO (in sequential order)
        processed_pos = []
        for idx, (po_bytes, po_filename, po_number, page_range, language_map) in enumerate(po_files, 1):
            try:
                # Determine original languages for this PO's pages
                # Extract page indices from page_range (e.g., "1-3" -> [0,1,2])
                start, end = map(int, page_range.split('-'))
                page_indices = list(range(start - 1, end))  # Convert to 0-indexed
                original_langs = list(set(language_map.get(i, 'unknown') for i in page_indices))
                
                po_info = process_single_po(po_bytes, po_filename, po_number, idx, 
                                            page_range, original_langs, dirs)
                
                processed_pos.append({
                    "sequence": idx,
                    "filename": po_info.filename,
                    "po_number": po_info.po_number,
                    "page_range": page_range,
                    "page_count": po_info.page_count,
                    "original_languages": original_langs,
                    "file_hash": po_info.file_hash,
                    "file_size_bytes": len(po_bytes)
                })
                logger.info(f"Successfully processed PO {idx}/{len(po_files)}")
            except Exception as e:
                logger.error(f"Error processing PO {po_filename}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                processed_pos.append({
                    "sequence": idx,
                    "filename": po_filename,
                    "page_range": page_range,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        result = SplitResult(
            original_filename=original_filename,
            total_pos_found=len(po_files),
            po_files=processed_pos,
            processing_time=processing_time
        )
        
        logger.info(f"Processing complete: {len(po_files)} POs in {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in split_and_process_pos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Multi-PO PDF Extraction"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
, value_content, re.IGNORECASE):
                        # This could be a PO number if the key suggests it
                        # Language-agnostic check: keys that are short and at top of document
                        if len(key_content) < 50:  # Keys are usually short
                            logger.info(f"Found potential PO number via KV pair: {value_content}")
                            return value_content
        
        # Strategy 3: Look for any field with strong alphanumeric pattern in top section
        if result.pages:
            first_page = result.pages[0]
            page_height = first_page.height if hasattr(first_page, 'height') else 1000
            threshold = page_height * 0.25  # Top 25% of page
            
            for line in first_page.lines if hasattr(first_page, 'lines') else []:
                if hasattr(line, 'polygon') and line.polygon and len(line.polygon) >= 2:
                    y_coord = line.polygon[0].y if hasattr(line.polygon[0], 'y') else 0
                    if y_coord < threshold:
                        content = line.content.strip()
                        # Look for standalone alphanumeric values
                        tokens = content.split()
                        for token in tokens:
                            if re.match(r'^[A-Z0-9]{5,20}


def group_pages_by_po(page_bytes_list: List[bytes]) -> List[Dict[str, Any]]:
    """
    Group pages by PO number. Pages with the same PO number are grouped together.
    Pages without a PO number are attached to the current group or form a new group.
    
    Returns: List of dicts with keys 'po_number' and 'pages'
    """
    groups = []
    current_group = []
    current_po_number = None
    
    logger.info(f"Starting to group {len(page_bytes_list)} pages by PO")
    
    for i, page_bytes in enumerate(page_bytes_list):
        po_num = extract_po_number(page_bytes)
        logger.info(f"Page {i+1}: PO number = {po_num if po_num else 'None'}")
        
        if po_num:
            # Found a PO number
            if current_po_number is None:
                # First PO found
                current_group = [page_bytes]
                current_po_number = po_num
            elif po_num == current_po_number:
                # Same PO, add to current group
                current_group.append(page_bytes)
            else:
                # Different PO, save current group and start new one
                groups.append({
                    'po_number': current_po_number,
                    'pages': current_group,
                    'page_count': len(current_group)
                })
                logger.info(f"Completed group: PO={current_po_number}, pages={len(current_group)}")
                current_group = [page_bytes]
                current_po_number = po_num
        else:
            # No PO number found on this page
            if not current_group:
                # No current group, start a new one without PO
                current_group = [page_bytes]
                current_po_number = None
            else:
                # Add to current group (continuation page)
                current_group.append(page_bytes)
    
    # Save last group
    if current_group:
        groups.append({
            'po_number': current_po_number,
            'pages': current_group,
            'page_count': len(current_group)
        })
        logger.info(f"Completed final group: PO={current_po_number}, pages={len(current_group)}")
    
    logger.info(f"Total PO groups created: {len(groups)}")
    return groups


def assemble_pdf_from_pages(page_bytes_group: List[bytes]) -> bytes:
    """
    Assemble multiple single-page PDFs into one multi-page PDF.
    """
    pdf = fitz.open()
    
    for idx, page_bytes in enumerate(page_bytes_group):
        try:
            single_doc = fitz.open(stream=page_bytes, filetype='pdf')
            pdf.insert_pdf(single_doc, from_page=0, to_page=0)
            single_doc.close()
        except Exception as e:
            logger.error(f"Error assembling page {idx}: {e}")
    
    out_bytes = pdf.write()
    pdf.close()
    return bytes(out_bytes)


def extract_pos_from_pdf(pdf_bytes: bytes, original_filename: str) -> List[Tuple[bytes, str, Optional[str]]]:
    """
    Main function to extract individual POs from a multi-PO PDF.
    
    Returns: List of tuples (pdf_bytes, filename, po_number)
    """
    logger.info(f"Starting PO extraction for: {original_filename}")
    
    # Step 1: Split into individual pages
    page_bytes_list = split_pdf_pages(pdf_bytes)
    if not page_bytes_list:
        logger.error("Failed to split PDF into pages")
        return []
    
    # Step 2: Group pages by PO number
    groups = group_pages_by_po(page_bytes_list)
    if not groups:
        logger.error("No PO groups found")
        return []
    
    # Step 3: Assemble each group into a separate PDF
    po_files = []
    base_name = os.path.splitext(original_filename)[0]
    
    for idx, group in enumerate(groups, 1):
        if not group['pages']:
            logger.warning(f"Group {idx} has no pages, skipping")
            continue
        
        try:
            # Assemble PDF from pages
            po_pdf_bytes = assemble_pdf_from_pages(group['pages'])
            
            # Generate filename
            if group['po_number']:
                po_filename = f"{base_name}_PO_{group['po_number']}.pdf"
            else:
                po_filename = f"{base_name}_PO_Unknown_{idx}.pdf"
            
            po_files.append((po_pdf_bytes, po_filename, group['po_number']))
            logger.info(f"Created PO file: {po_filename} ({len(group['pages'])} pages)")
            
        except Exception as e:
            logger.error(f"Error creating PO file for group {idx}: {e}")
    
    logger.info(f"Successfully extracted {len(po_files)} PO files from {original_filename}")
    return po_files


# ============================================================================
# FILE PROCESSING AND EXTRACTION
# ============================================================================

def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


def get_user_directories(user_directory: str) -> Dict[str, str]:
    """Get or create user-specific directories."""
    sanitized = re.sub(r'\W+', '_', user_directory)
    base = os.path.join(ROOT_DIR, "user_data", sanitized)
    upload = os.path.join(base, "PO_data")
    extracted = os.path.join(base, "Extracted_data")
    seq_file = os.path.join(base, "sequence_mapping.json")
    
    for d in [upload, extracted]:
        os.makedirs(d, exist_ok=True)
    
    return {
        "base": base,
        "upload": upload,
        "extracted": extracted,
        "sequence_file": seq_file
    }


def save_po_file(file_content: bytes, filename: str, dirs: Dict[str, str]) -> str:
    """Save PO file to user's upload directory."""
    file_path = os.path.join(dirs["upload"], filename)
    with open(file_path, 'wb') as f:
        f.write(file_content)
    logger.info(f"Saved PO file: {file_path}")
    return file_path


def extract_po_data(file_content: bytes, filename: str, dirs: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract structured data from a PO using Azure Document Intelligence.
    """
    extract_file_path = os.path.join(dirs["extracted"], f"{os.path.splitext(filename)[0]}_extracted.json")
    
    # Check if already extracted
    if os.path.exists(extract_file_path):
        try:
            with open(extract_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading existing extracted file: {e}")
    
    try:
        # Analyze document with Azure DI
        poller = document_client.begin_analyze_document(
            "prebuilt-document",  # Or use custom PO model
            file_content
        )
        result = poller.result()
        
        # Extract fields
        extracted_fields = {}
        for doc in result.documents:
            for field_name, field_value in doc.fields.items():
                if hasattr(field_value, 'value') and field_value.value is not None:
                    extracted_fields[field_name] = field_value.value
        
        # Extract tables
        extracted_tables = []
        for table_idx, table in enumerate(result.tables):
            table_data = []
            for row_idx in range(table.row_count):
                row_data = []
                for col_idx in range(table.column_count):
                    cell_text = ""
                    for cell in table.cells:
                        if cell.row_index == row_idx and cell.column_index == col_idx:
                            cell_text = cell.content
                            break
                    row_data.append(cell_text)
                table_data.append(row_data)
            
            extracted_tables.append({
                "table_index": table_idx,
                "data": table_data,
                "row_count": table.row_count,
                "column_count": table.column_count
            })
        
        # Extract key-value pairs
        key_value_pairs = []
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.value:
                key_value_pairs.append({
                    "key": kv_pair.key.content,
                    "value": kv_pair.value.content
                })
        
        extracted_data = {
            "fields": extracted_fields,
            "tables": extracted_tables,
            "key_value_pairs": key_value_pairs,
            "document_type": result.documents[0].doc_type if result.documents else "Unknown"
        }
        
        # Save extracted data
        with open(extract_file_path, 'w') as f:
            json.dump(extracted_data, f, default=str, indent=2)
        
        logger.info(f"Extracted data saved: {extract_file_path}")
        return extracted_data
        
    except Exception as e:
        logger.error(f"Error extracting PO data: {e}")
        return {"error": str(e)}


def process_single_po(po_bytes: bytes, po_filename: str, po_number: Optional[str], 
                      sequence_num: int, dirs: Dict[str, str]) -> POInfo:
    """Process a single PO file and return POInfo."""
    
    # Save file
    save_po_file(po_bytes, po_filename, dirs)
    
    # Calculate hash
    file_hash = calculate_file_hash(po_bytes)
    
    # Extract data
    extracted_data = extract_po_data(po_bytes, po_filename, dirs)
    
    # Count pages
    try:
        doc = fitz.open(stream=po_bytes, filetype="pdf")
        page_count = doc.page_count
        doc.close()
    except:
        page_count = 0
    
    return POInfo(
        filename=po_filename,
        po_number=po_number,
        extracted_data=extracted_data,
        page_count=page_count,
        processed_at=time.time(),
        file_hash=file_hash,
        sequence_number=sequence_num
    )


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Multi-PO PDF Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/split-and-process-pos")
async def split_and_process_pos(
    file: UploadFile = File(...),
    user_id: str = "default_user"
):
    """
    Upload a multi-PO PDF and split it into individual PO files.
    Each PO is processed and data is extracted.
    """
    start_time = time.time()
    
    try:
        # Read uploaded file
        pdf_bytes = await file.read()
        original_filename = file.filename
        
        logger.info(f"Received file: {original_filename}, size: {len(pdf_bytes)} bytes")
        
        # Get user directories
        dirs = get_user_directories(user_id)
        
        # Extract POs from PDF
        po_files = extract_pos_from_pdf(pdf_bytes, original_filename)
        
        if not po_files:
            raise HTTPException(status_code=400, detail="No POs found in PDF")
        
        # Process each PO
        processed_pos = []
        for idx, (po_bytes, po_filename, po_number) in enumerate(po_files, 1):
            try:
                po_info = process_single_po(po_bytes, po_filename, po_number, idx, dirs)
                processed_pos.append({
                    "filename": po_info.filename,
                    "po_number": po_info.po_number,
                    "page_count": po_info.page_count,
                    "file_hash": po_info.file_hash,
                    "sequence_number": po_info.sequence_number
                })
            except Exception as e:
                logger.error(f"Error processing PO {po_filename}: {e}")
                processed_pos.append({
                    "filename": po_filename,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        result = SplitResult(
            original_filename=original_filename,
            total_pos_found=len(po_files),
            po_files=processed_pos,
            processing_time=processing_time
        )
        
        logger.info(f"Processing complete: {len(po_files)} POs in {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in split_and_process_pos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Multi-PO PDF Extraction"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
, token, re.IGNORECASE):
                                # Strong candidate for PO number
                                logger.info(f"Found PO number via pattern match: {token}")
                                return token
        
    except Exception as e:
        logger.warning(f"Azure DI extraction failed: {e}")
    
    return None


def group_pages_by_po(page_bytes_list: List[bytes]) -> List[Dict[str, Any]]:
    """
    Group pages by PO number. Pages with the same PO number are grouped together.
    Pages without a PO number are attached to the current group or form a new group.
    
    Returns: List of dicts with keys 'po_number' and 'pages'
    """
    groups = []
    current_group = []
    current_po_number = None
    
    logger.info(f"Starting to group {len(page_bytes_list)} pages by PO")
    
    for i, page_bytes in enumerate(page_bytes_list):
        po_num = extract_po_number(page_bytes)
        logger.info(f"Page {i+1}: PO number = {po_num if po_num else 'None'}")
        
        if po_num:
            # Found a PO number
            if current_po_number is None:
                # First PO found
                current_group = [page_bytes]
                current_po_number = po_num
            elif po_num == current_po_number:
                # Same PO, add to current group
                current_group.append(page_bytes)
            else:
                # Different PO, save current group and start new one
                groups.append({
                    'po_number': current_po_number,
                    'pages': current_group,
                    'page_count': len(current_group)
                })
                logger.info(f"Completed group: PO={current_po_number}, pages={len(current_group)}")
                current_group = [page_bytes]
                current_po_number = po_num
        else:
            # No PO number found on this page
            if not current_group:
                # No current group, start a new one without PO
                current_group = [page_bytes]
                current_po_number = None
            else:
                # Add to current group (continuation page)
                current_group.append(page_bytes)
    
    # Save last group
    if current_group:
        groups.append({
            'po_number': current_po_number,
            'pages': current_group,
            'page_count': len(current_group)
        })
        logger.info(f"Completed final group: PO={current_po_number}, pages={len(current_group)}")
    
    logger.info(f"Total PO groups created: {len(groups)}")
    return groups


def assemble_pdf_from_pages(page_bytes_group: List[bytes]) -> bytes:
    """
    Assemble multiple single-page PDFs into one multi-page PDF.
    """
    pdf = fitz.open()
    
    for idx, page_bytes in enumerate(page_bytes_group):
        try:
            single_doc = fitz.open(stream=page_bytes, filetype='pdf')
            pdf.insert_pdf(single_doc, from_page=0, to_page=0)
            single_doc.close()
        except Exception as e:
            logger.error(f"Error assembling page {idx}: {e}")
    
    out_bytes = pdf.write()
    pdf.close()
    return bytes(out_bytes)


def extract_pos_from_pdf(pdf_bytes: bytes, original_filename: str) -> List[Tuple[bytes, str, Optional[str]]]:
    """
    Main function to extract individual POs from a multi-PO PDF.
    
    Returns: List of tuples (pdf_bytes, filename, po_number)
    """
    logger.info(f"Starting PO extraction for: {original_filename}")
    
    # Step 1: Split into individual pages
    page_bytes_list = split_pdf_pages(pdf_bytes)
    if not page_bytes_list:
        logger.error("Failed to split PDF into pages")
        return []
    
    # Step 2: Group pages by PO number
    groups = group_pages_by_po(page_bytes_list)
    if not groups:
        logger.error("No PO groups found")
        return []
    
    # Step 3: Assemble each group into a separate PDF
    po_files = []
    base_name = os.path.splitext(original_filename)[0]
    
    for idx, group in enumerate(groups, 1):
        if not group['pages']:
            logger.warning(f"Group {idx} has no pages, skipping")
            continue
        
        try:
            # Assemble PDF from pages
            po_pdf_bytes = assemble_pdf_from_pages(group['pages'])
            
            # Generate filename
            if group['po_number']:
                po_filename = f"{base_name}_PO_{group['po_number']}.pdf"
            else:
                po_filename = f"{base_name}_PO_Unknown_{idx}.pdf"
            
            po_files.append((po_pdf_bytes, po_filename, group['po_number']))
            logger.info(f"Created PO file: {po_filename} ({len(group['pages'])} pages)")
            
        except Exception as e:
            logger.error(f"Error creating PO file for group {idx}: {e}")
    
    logger.info(f"Successfully extracted {len(po_files)} PO files from {original_filename}")
    return po_files


# ============================================================================
# FILE PROCESSING AND EXTRACTION
# ============================================================================

def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


def get_user_directories(user_directory: str) -> Dict[str, str]:
    """Get or create user-specific directories."""
    sanitized = re.sub(r'\W+', '_', user_directory)
    base = os.path.join(ROOT_DIR, "user_data", sanitized)
    upload = os.path.join(base, "PO_data")
    extracted = os.path.join(base, "Extracted_data")
    seq_file = os.path.join(base, "sequence_mapping.json")
    
    for d in [upload, extracted]:
        os.makedirs(d, exist_ok=True)
    
    return {
        "base": base,
        "upload": upload,
        "extracted": extracted,
        "sequence_file": seq_file
    }


def save_po_file(file_content: bytes, filename: str, dirs: Dict[str, str]) -> str:
    """Save PO file to user's upload directory."""
    file_path = os.path.join(dirs["upload"], filename)
    with open(file_path, 'wb') as f:
        f.write(file_content)
    logger.info(f"Saved PO file: {file_path}")
    return file_path


def extract_po_data(file_content: bytes, filename: str, dirs: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract structured data from a PO using Azure Document Intelligence.
    """
    extract_file_path = os.path.join(dirs["extracted"], f"{os.path.splitext(filename)[0]}_extracted.json")
    
    # Check if already extracted
    if os.path.exists(extract_file_path):
        try:
            with open(extract_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading existing extracted file: {e}")
    
    try:
        # Analyze document with Azure DI
        poller = document_client.begin_analyze_document(
            "prebuilt-document",  # Or use custom PO model
            file_content
        )
        result = poller.result()
        
        # Extract fields
        extracted_fields = {}
        for doc in result.documents:
            for field_name, field_value in doc.fields.items():
                if hasattr(field_value, 'value') and field_value.value is not None:
                    extracted_fields[field_name] = field_value.value
        
        # Extract tables
        extracted_tables = []
        for table_idx, table in enumerate(result.tables):
            table_data = []
            for row_idx in range(table.row_count):
                row_data = []
                for col_idx in range(table.column_count):
                    cell_text = ""
                    for cell in table.cells:
                        if cell.row_index == row_idx and cell.column_index == col_idx:
                            cell_text = cell.content
                            break
                    row_data.append(cell_text)
                table_data.append(row_data)
            
            extracted_tables.append({
                "table_index": table_idx,
                "data": table_data,
                "row_count": table.row_count,
                "column_count": table.column_count
            })
        
        # Extract key-value pairs
        key_value_pairs = []
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.value:
                key_value_pairs.append({
                    "key": kv_pair.key.content,
                    "value": kv_pair.value.content
                })
        
        extracted_data = {
            "fields": extracted_fields,
            "tables": extracted_tables,
            "key_value_pairs": key_value_pairs,
            "document_type": result.documents[0].doc_type if result.documents else "Unknown"
        }
        
        # Save extracted data
        with open(extract_file_path, 'w') as f:
            json.dump(extracted_data, f, default=str, indent=2)
        
        logger.info(f"Extracted data saved: {extract_file_path}")
        return extracted_data
        
    except Exception as e:
        logger.error(f"Error extracting PO data: {e}")
        return {"error": str(e)}


def process_single_po(po_bytes: bytes, po_filename: str, po_number: Optional[str], 
                      sequence_num: int, dirs: Dict[str, str]) -> POInfo:
    """Process a single PO file and return POInfo."""
    
    # Save file
    save_po_file(po_bytes, po_filename, dirs)
    
    # Calculate hash
    file_hash = calculate_file_hash(po_bytes)
    
    # Extract data
    extracted_data = extract_po_data(po_bytes, po_filename, dirs)
    
    # Count pages
    try:
        doc = fitz.open(stream=po_bytes, filetype="pdf")
        page_count = doc.page_count
        doc.close()
    except:
        page_count = 0
    
    return POInfo(
        filename=po_filename,
        po_number=po_number,
        extracted_data=extracted_data,
        page_count=page_count,
        processed_at=time.time(),
        file_hash=file_hash,
        sequence_number=sequence_num
    )


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Multi-PO PDF Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/split-and-process-pos")
async def split_and_process_pos(
    file: UploadFile = File(...),
    user_id: str = "default_user"
):
    """
    Upload a multi-PO PDF and split it into individual PO files.
    Each PO is processed and data is extracted.
    """
    start_time = time.time()
    
    try:
        # Read uploaded file
        pdf_bytes = await file.read()
        original_filename = file.filename
        
        logger.info(f"Received file: {original_filename}, size: {len(pdf_bytes)} bytes")
        
        # Get user directories
        dirs = get_user_directories(user_id)
        
        # Extract POs from PDF
        po_files = extract_pos_from_pdf(pdf_bytes, original_filename)
        
        if not po_files:
            raise HTTPException(status_code=400, detail="No POs found in PDF")
        
        # Process each PO
        processed_pos = []
        for idx, (po_bytes, po_filename, po_number) in enumerate(po_files, 1):
            try:
                po_info = process_single_po(po_bytes, po_filename, po_number, idx, dirs)
                processed_pos.append({
                    "filename": po_info.filename,
                    "po_number": po_info.po_number,
                    "page_count": po_info.page_count,
                    "file_hash": po_info.file_hash,
                    "sequence_number": po_info.sequence_number
                })
            except Exception as e:
                logger.error(f"Error processing PO {po_filename}: {e}")
                processed_pos.append({
                    "filename": po_filename,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        result = SplitResult(
            original_filename=original_filename,
            total_pos_found=len(po_files),
            po_files=processed_pos,
            processing_time=processing_time
        )
        
        logger.info(f"Processing complete: {len(po_files)} POs in {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in split_and_process_pos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Multi-PO PDF Extraction"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
