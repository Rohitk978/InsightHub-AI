import pdfplumber
import logging
from chatbot import chat_with_text  # Assumes this function exists in chatbot.py
from summarization import summarize_text  # Assumes this function exists in summaraization.py

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using pdfplumber.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from all pages of the PDF.
    
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If no text is extracted from the PDF.
        Exception: For other errors during extraction.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if not text.strip():
                raise ValueError("No text extracted from the PDF.")
            logger.info(f"Extracted text from {len(pdf.pages)} pages in {pdf_path}")
            return text
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        raise
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        raise

def process_extracted_text(text):
    """
    Process the extracted text by feeding it to the chatbot and summarization models.
    
    Args:
        text (str): Text extracted from the PDF.
    
    Returns:
        tuple: (chatbot_response, summary) from the respective models.
    
    Raises:
        Exception: For errors during processing.
    """
    try:
        chatbot_response = chat_with_text(text)
        summary = summarize_text(text)
        logger.info("Text successfully processed by chatbot and summarization models")
        return chatbot_response, summary
    except Exception as e:
        logger.error(f"Error processing extracted text: {str(e)}")
        raise

def extract_and_process(pdf_path):
    """
    Extract text from a PDF and feed it to the chatbot and summarization models.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        tuple: (chatbot_response, summary) from the processed text.
    """
    text = extract_text_from_pdf(pdf_path)
    return process_extracted_text(text)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pdf_extractor.py <pdf_path>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    try:
        chatbot_response, summary = extract_and_process(pdf_path)
        print("Chatbot Response:", chatbot_response)
        print("Summary:", summary)
    except Exception as e:
        print(f"Error: {str(e)}")