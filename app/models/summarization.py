from .utils import load_model, load_tokenizer, chunk_text
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import os
import logging

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.max_words = 10000  # Reasonable limit to prevent CUDA errors
        self.max_file_size_mb = 5
        self.max_input_tokens = 1024  # BART's max token limit

    def _load_model(self):
        if self.model is None:
            self.model = load_model("summarization", self.model_name)
            self.tokenizer = load_tokenizer(self.model_name)

    def summarize_text(self, text, max_length=150, min_length=40):
        self._load_model()
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Truncate text to max_words
        words = text.split()
        if len(words) > self.max_words:
            text = ' '.join(words[:self.max_words])
            logger.warning(f"Text truncated to {self.max_words} words")
        
        # Count tokens and adjust max_length
        input_tokens = self.tokenizer.encode(text, max_length=self.max_input_tokens, truncation=True)
        input_length = len(input_tokens)
        max_length = min(max_length, max(input_length // 2, min_length))
        
        # Chunk text into smaller pieces
        chunks = chunk_text(text, max_tokens=512, tokenizer=self.tokenizer)
        summaries = []
        
        try:
            for chunk in chunks[:10]:  # Limit to 10 chunks to avoid OOM
                try:
                    summary = self.model(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                        truncation=True
                    )
                    summaries.append(summary[0]['summary_text'])
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        logger.error(f"CUDA error in chunk summarization: {str(e)}")
                        raise ValueError(f"GPU error during summarization: {str(e)}")
                    else:
                        logger.warning(f"Error summarizing chunk, skipping: {str(e)}")
                        continue
            if not summaries:
                raise ValueError("No summaries generated")
            return ' '.join(summaries)
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            raise ValueError(f"Summarization failed: {str(e)}")

    def summarize_url(self, url, max_length=150, min_length=40):
        try:
            response = requests.get(url, timeout=10, headers={'User-Agent': os.getenv('USER_AGENT', 'ai-qa-project/1.0')})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text_elements = soup.find_all(['h1', 'h2', 'h3', 'p', 'li'])
            text = ' '.join(elem.get_text(strip=True) for elem in text_elements if elem.get_text(strip=True))
            if not text:
                raise ValueError("No meaningful text extracted from URL")
            logger.info(f"Extracted {len(text.split())} words from URL: {url}")
            return self.summarize_text(text, max_length, min_length)
        except requests.RequestException as e:
            logger.error(f"Failed to fetch URL {url}: {str(e)}")
            raise ValueError(f"Failed to fetch URL: {str(e)}")

    def summarize_pdf(self, filepath, max_length=150, min_length=40):
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"PDF too large ({file_size_mb:.2f}MB). Max allowed: {self.max_file_size_mb}MB")
        
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        if not text:
            raise ValueError("No text extracted from PDF")
        
        return self.summarize_text(text, max_length, min_length)