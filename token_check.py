from huggingface_hub import InferenceClient, HfApi
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print("Token:", token[:5] + "..." if token else "No token")

# Verify token using HfApi
try:
    api = HfApi(token=token)
    user_info = api.whoami()
    print("Authenticated as:", user_info["name"])
except Exception as auth_e:
    print("Token validation failed:", str(auth_e))
    exit(1)

# Test model access
try:
    client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=token, provider="hf-inference")
    response = client.text_generation("Test prompt", max_new_tokens=10)
    print("Model response:", response)
except Exception as e:
    print("Model access failed:", str(e))