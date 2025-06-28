import os
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, GPT2LMHeadModel, GPT2Tokenizer
import torch
from flask import current_app
import logging
import shutil
import time

logger = logging.getLogger(__name__)

# Set Hugging Face cache directory
if not os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = "D:\\HuggingFaceCache"

# Download NLTK data
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
punkt_path = os.path.join(nltk_data_dir, "tokenizers/punkt")
punkt_tab_path = os.path.join(nltk_data_dir, "tokenizers/punkt/PY3/english.pickle")

if not os.path.exists(punkt_path):
    nltk.download('punkt', quiet=True)
if not os.path.exists(punkt_tab_path):
    nltk.download('punkt_tab', quiet=True)

_models = {}
_tokenizers = {}
_pipelines = {}

def load_model(task, model_name, max_retries=2):
    if model_name not in _pipelines:
        cache_dir = os.path.join(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub", f"models--{model_name.replace('/', '--')}")
        for attempt in range(max_retries):
            try:
                # Check available VRAM
                device = -1  # Default to CPU
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
                        used_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
                        if vram - used_vram > 1.0:  # Require ~1GB free for distilgpt2
                            device = 0
                    except Exception as e:
                        logger.warning(f"VRAM check failed: {str(e)}. Falling back to CPU.")
                
                logger.info(f"Loading model {model_name} on device {'cuda:0' if device == 0 else 'cpu'} (Attempt {attempt + 1}/{max_retries})")
                token = current_app.config.get('HUGGINGFACEHUB_API_TOKEN')
                
                # Check for corrupted cache
                if os.path.exists(cache_dir):
                    pytorch_bin = os.path.join(cache_dir, "pytorch_model.bin")
                    if not os.path.exists(pytorch_bin):
                        logger.warning(f"No pytorch_model.bin found in {cache_dir}. Clearing cache.")
                        shutil.rmtree(cache_dir, ignore_errors=True)
                
                if task == "gpt2":
                    model = GPT2LMHeadModel.from_pretrained(model_name, token=token)
                    tokenizer = GPT2Tokenizer.from_pretrained(model_name, token=token)
                    # Set pad token
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token = tokenizer.eos_token
                        model.config.pad_token_id = tokenizer.eos_token_id
                    # Move model to device
                    model = model.to(torch.device("cuda:0" if device == 0 else "cpu"))
                    _models[model_name] = model
                    _tokenizers[model_name] = tokenizer
                    _pipelines[model_name] = model
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token, from_flax=False)
                    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
                    model = model.to(torch.device("cuda:0" if device == 0 else "cpu"))
                    _models[model_name] = model
                    _tokenizers[model_name] = tokenizer
                    _pipelines[model_name] = pipeline(
                        task,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        framework="pt",
                        truncation=True,
                        max_length=1024
                    )
                logger.info(f"Successfully loaded model {model_name}")
                break
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)} on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying download for {model_name}...")
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    time.sleep(1)
                else:
                    raise
    return _pipelines[model_name]

def load_tokenizer(model_name):
    if model_name not in _tokenizers:
        try:
            token = current_app.config.get('HUGGINGFACEHUB_API_TOKEN')
            logger.info(f"Loading tokenizer for {model_name}")
            if "gpt2" in model_name:
                tokenizer = GPT2Tokenizer.from_pretrained(model_name, token=token)
                # Set pad token
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                _tokenizers[model_name] = tokenizer
            else:
                _tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, token=token)
            logger.info(f"Successfully loaded tokenizer for {model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {str(e)}")
            raise
    return _tokenizers[model_name]

def chunk_text(text, max_tokens=512, tokenizer=None):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        token_count = len(tokenizer.encode(sentence, max_length=512, truncation=True) if tokenizer else sentence.split())
        if current_length + token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = token_count
        else:
            current_chunk.append(sentence)
            current_length += token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks