from flask import request, current_app
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import pdfplumber
import os
from .validators import validate_file_extension

# def handle_pdf_upload(file_key):
#     """
#     Handle PDF file upload and return the filepath.
    
#     Args:
#         file_key (str): Form key for the file.
    
#     Returns:
#         str: Filepath of the saved file, or (error_message, status_code) if invalid.
#     """
#     if file_key not in request.files:
#         return "No file uploaded", 400
#     file = request.files[file_key]
#     if file.filename == '':
#         return "No file selected", 400
#     if not validate_file_extension(file.filename, ['.pdf']):
#         return "Only PDF files are supported", 400
    
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(request.app.config['UPLOAD_FOLDER'], filename)
#     try:
#         file.save(filepath)
#         file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
#         if file_size_mb > request.app.config.get('MAX_CONTENT_LENGTH', 16) / (1024 * 1024):
#             os.remove(filepath)
#             return f"PDF too large ({file_size_mb:.2f}MB). Max allowed: 16MB", 400
#         return filepath
#     except Exception as e:
#         if os.path.exists(filepath):
#             os.remove(filepath)
#         return f"Failed to save PDF: {e}", 500

def handle_pdf_upload(request, file_key):
    """
    Handle PDF file upload and return the filepath.
    
    Args:
        request: Flask request object.
        file_key (str): Form key for the file.
    
    Returns:
        str: Filepath of the saved file, or (error_message, status_code) if invalid.
    """
    if file_key not in request.files:
        return "No file uploaded", 400
    file = request.files[file_key]
    if file.filename == '':
        return "No file selected", 400
    if not validate_file_extension(file.filename, ['.pdf']):
        return "Only PDF files are supported", 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(filepath)
        file_size_bytes = os.path.getsize(filepath)
        max_size_bytes = current_app.config.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)  # Default 16MB
        if file_size_bytes > max_size_bytes:
            os.remove(filepath)
            return f"PDF too large ({file_size_bytes / (1024 * 1024):.2f}MB). Max allowed: {max_size_bytes / (1024 * 1024)}MB", 400
        return filepath
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return f"Failed to save PDF: {str(e)}", 500
def handle_file_upload(file_key):
    """
    Handle file upload (PDF or text) and extract content.
    
    Args:
        file_key (str): Form key for the file.
    
    Returns:
        str: Extracted text, or (error_message, status_code) if invalid.
    """
    if file_key not in request.files:
        return "No file uploaded", 400
    file = request.files[file_key]
    if file.filename == '':
        return "No file selected", 400
    if not validate_file_extension(file.filename, ['.pdf', '.txt']):
        return "Unsupported file type. Use .pdf or .txt", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(request.app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath  # Simplified for example; add extraction logic as needed