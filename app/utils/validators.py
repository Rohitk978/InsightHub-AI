import re

def validate_text_input(text, max_length=5000):
    """
    Validate text input for emptiness and length.
    
    Args:
        text (str): Input text to validate.
        max_length (int): Maximum allowed length in characters.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    if len(text) > max_length:
        return False
    return True

def validate_url(url):
    """
    Validate URL format.
    
    Args:
        url (str): URL to validate.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    url_pattern = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')
    return isinstance(url, str) and bool(url_pattern.match(url))

def validate_file_extension(filename, allowed_extensions=['.pdf', '.txt']):
    """
    Validate file extension.
    
    Args:
        filename (str): Name of the file.
        allowed_extensions (list): List of allowed file extensions.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    return isinstance(filename, str) and any(filename.lower().endswith(ext) for ext in allowed_extensions)