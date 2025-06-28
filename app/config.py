import os
from dotenv import load_dotenv
import torch

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "default-secret")
    FLASK_ENV = os.getenv("FLASK_ENV", "production")
    # UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "Uploads")
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MODEL_PATH = os.getenv("MODEL_PATH", "./models")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    REPO_ID = os.getenv("REPO_ID", "meta-llama/Llama-3.1-8B-Instruct")
    USER_AGENT = os.getenv("USER_AGENT", "ai-qa-project/1.0")
    CUDA_AVAILABLE = torch.cuda.is_available()
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB in bytes