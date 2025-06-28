# from flask import Blueprint, request, render_template
# from ..models.chatbot import DocumentQA
# from ..utils.validators import validate_text_input
# import logging
# import os
# from werkzeug.utils import secure_filename

# chatbot_bp = Blueprint("chatbot", __name__)
# logger = logging.getLogger(__name__)

# ALLOWED_EXTENSIONS = {'pdf'}
# UPLOAD_FOLDER = os.path.join('Uploads')

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @chatbot_bp.route("/", methods=["POST"])
# def chat():
#     try:
#         context_type = request.form.get("context_type")
#         question = request.form.get("question")
#         if not validate_text_input(question):
#             logger.error("Invalid or empty question")
#             return render_template("error.html", error="Invalid or empty question"), 400

#         if context_type == "text":
#             context = request.form.get("context_text")
#             if not validate_text_input(context):
#                 logger.error("Invalid or empty context text")
#                 return render_template("error.html", error="Invalid or empty context text"), 400
#             chatbot = DocumentQA(document=context, is_file=False)
#         elif context_type == "file":
#             if 'context_file' not in request.files:
#                 logger.error("No file uploaded")
#                 return render_template("error.html", error="No PDF file uploaded"), 400
            
#             file = request.files['context_file']
#             if file.filename == '':
#                 logger.error("No file selected")
#                 return render_template("error.html", error="No PDF file selected"), 400
            
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
#                 os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#                 file_path = os.path.join(UPLOAD_FOLDER, filename)
#                 try:
#                     file.save(file_path)
#                 except Exception as e:
#                     logger.error(f"Failed to save file: {str(e)}")
#                     return render_template("error.html", error="Failed to save PDF file."), 400
                
#                 try:
#                     chatbot = DocumentQA(document=file_path, is_file=True)
#                 finally:
#                     try:
#                         os.remove(file_path)
#                     except Exception as e:
#                         logger.warning(f"Failed to delete file {file_path}: {str(e)}")
#             else:
#                 logger.error("Invalid file type")
#                 return render_template("error.html", error="Only PDF files are allowed"), 400
#         else:
#             logger.error("Invalid context type")
#             return render_template("error.html", error="Invalid context type"), 400

#         answer = chatbot.answer_query(question)
#         logger.info(f"Chat response generated for question: {question[:50]}...")
#         return render_template("result.html", result=answer["result"], sources=[doc.page_content for doc in answer.get("source_documents", [])], input_type="chat")
    
#     except Exception as e:
#         logger.error(f"Error in chat: {str(e)}")
#         return render_template("error.html", error=f"An error occurred: {str(e)}"), 500

# @chatbot_bp.route("/", methods=["GET"])
# def chat_form():
#     return render_template("chat.html")






























# 2





# from flask import Blueprint, request, render_template
# import logging
# import traceback
# from dotenv import load_dotenv
# import os
# from huggingface_hub import InferenceClient
# import pdfplumber
# from io import BytesIO

# # Load environment variables from .env file
# load_dotenv()

# chatbot_bp = Blueprint("chatbot", __name__)
# logger = logging.getLogger(__name__)

# @chatbot_bp.route("/chat", methods=["GET", "POST"])
# def chat():
#     if request.method == "GET":
#         return render_template("chat.html")
    
#     try:
#         # Validate token
#         token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#         if not token:
#             raise ValueError("HUGGINGFACE_TOKEN not found in .env file.")
        
#         # # Verify token authenticity
#         # try:
#         #     user = who_am_i(token=token)
#         #     logger.info(f"Authenticated as {user['name']}")
#         # except Exception as auth_e:
#         #     raise ValueError(f"Invalid Hugging Face token: {str(auth_e)}")
        
#         # Get input: PDF or text
#         if 'pdf_file' in request.files and request.files['pdf_file'].filename:
#             pdf_file = request.files['pdf_file']
#             if not pdf_file.filename.lower().endswith('.pdf'):
#                 return render_template("error.html", error="Invalid file format. Please upload a PDF."), 400
#             # Extract text from PDF using pdfplumber
#             with pdfplumber.open(BytesIO(pdf_file.read())) as pdf:
#                 document = ""
#                 for page in pdf.pages:
#                     document += (page.extract_text() or "") + "\n"
#             is_pdf = True
#         elif 'context_text' in request.form and request.form['context_text'].strip():
#             document = request.form['context_text']
#             is_pdf = False
#         else:
#             return render_template("error.html", error="Please provide either a PDF file or text context."), 400
        
#         question = request.form.get("question")
#         if not question or not question.strip():
#             return render_template("error.html", error="Question is required."), 400
        
#         # Initialize InferenceClient
#         client = InferenceClient(
#             model="meta-llama/Llama-3.1-8B-Instruct",
#             token=token
#             # provider="hf-inference"
#         )
        
#         # Construct prompt
#         prompt = f"Context: {document}\nQuestion: {question}\nAnswer:"
        
#         # Query the model
#         answer = client.text_generation(
#             prompt,
#             max_new_tokens=512,
#             temperature=0.7,
#             top_p=0.9,
#             return_full_text=False
#         )
        
#         # Format response
#         response = {
#             "result": answer,
#             "source_documents": [{"page_content": document}] if is_pdf else []
#         }
        
#         return render_template(
#             "result.html",
#             result=response["result"],
#             sources=[doc["page_content"] for doc in response.get("source_documents", [])],
#             input_type="chat"
#         )
    
#     except Exception as e:
#         error_msg = str(e)
#         if "401" in error_msg or "403" in error_msg:
#             error_msg += " Authentication failed. Verify your token at https://huggingface.co/settings/tokens and ensure access to meta-llama/Llama-3.1-8B-Instruct at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct."
#         elif "404" in error_msg:
#             error_msg += " Model not found. Confirm the model ID or check Hugging Face Inference API status."
#         elif "Provider" in error_msg:
#             error_msg += " Check provider settings at https://hf.co/settings/inference-providers."
#         logger.error(f"Error in chat: {error_msg}\n{traceback.format_exc()}")
#         return render_template(
#             "error.html",
#             error=f"An error occurred: {error_msg}"
#         ), 500



import os
import logging
import traceback
from flask import Blueprint, request, render_template
from io import BytesIO
import pdfplumber
from huggingface_hub import InferenceClient
from ..utils.file_handler import handle_pdf_upload

# Suppress pdfplumber and pikepdf debug logging
# Suppress debug logging for pdfplumber, pikepdf, and related modules
for logger_name in logging.root.manager.loggerDict:
    if logger_name.startswith(('pdfplumber', 'pikepdf')):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

chatbot_bp = Blueprint("chatbot", __name__)
logger = logging.getLogger(__name__)


@chatbot_bp.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return render_template("chat.html")
    
    try:
        # Validate token
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN not found in .env file.")
        
        # Get input: PDF or text
        if request.form.get("context_type") == "file" and 'pdf_file' in request.files:
            filepath = handle_pdf_upload(request, "pdf_file")
            if isinstance(filepath, tuple):
                logger.error(f"PDF upload error: {filepath[0]}")
                return render_template("error.html", error=filepath[0]), 400
            # Extract text from PDF using pdfplumber
            try:
                with pdfplumber.open(filepath) as pdf:
                    document = ""
                    for page in pdf.pages:
                        document += (page.extract_text() or "") + "\n"
                is_pdf = True
                # Clean up the saved file
                os.remove(filepath)
            except Exception as e:
                logger.error(f"Failed to extract text from PDF: {e}")
                return render_template("error.html", error=f"Failed to process PDF: {str(e)}"), 400
        elif request.form.get("context_type") == "text" and request.form.get("context_text", "").strip():
            document = request.form["context_text"]
            is_pdf = False
        else:
            return render_template("error.html", error="Please provide either a PDF file or text context."), 400
        
        question = request.form.get("question")
        if not question or not question.strip():
            return render_template("error.html", error="Question is required."), 400
        
        # Initialize InferenceClient
        client = InferenceClient(
            model="meta-llama/Llama-3.1-8B-Instruct",
            token=token,
            provider="hf-inference"
        )
        
        # Construct prompt
        prompt = f"Context: {document}\nQuestion: {question}\nAnswer:"
        
        # Query the model
        answer = client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False
        )
        
        # Format response
        response = {
            "result": answer,
            "source_documents": [{"page_content": document}] if is_pdf else []
        }
        
        return render_template(
            "result.html",
            result=response["result"],
            sources=[doc["page_content"] for doc in response.get("source_documents", [])],
            input_type="chat"
        )
    
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "403" in error_msg:
            error_msg += " Authentication failed. Verify your token at https://huggingface.co/settings/tokens and ensure access to meta-llama/Llama-3.1-8B-Instruct."
        elif "404" in error_msg:
            error_msg += " Model not found. Confirm the model ID or check Hugging Face Inference API status."
        logger.error(f"Error in chat: {error_msg}\n{traceback.format_exc()}")
        return render_template(
            "error.html",
            error=f"An error occurred: {error_msg}"
        ), 500



























# 3
# from flask import Blueprint, request, render_template
# import logging
# import traceback
# from dotenv import load_dotenv
# import os
# from huggingface_hub import InferenceClient
# import pdfplumber
# from io import BytesIO

# # Load environment variables from .env file
# load_dotenv()

# chatbot_bp = Blueprint("chatbot", __name__)
# logger = logging.getLogger(__name__)

# @chatbot_bp.route("/", methods=["GET", "POST"])
# def chat():
#     if request.method == "GET":
#         return render_template("chat.html")
    
#     logger.info(f"Request files: {request.files}")
#     logger.info(f"Request form: {request.form}")
    
#     if 'pdf_file' in request.files and request.files['pdf_file'].filename:
#         pdf_file = request.files['pdf_file']
#         if not pdf_file.filename.lower().endswith('.pdf'):
#             return render_template("error.html", error="Invalid file format. Please upload a PDF."), 400
#         with pdfplumber.open(BytesIO(pdf_file.read())) as pdf:
#             document = ""
#             for page in pdf.pages:
#                 document += (page.extract_text() or "") + "\n"
#         is_pdf = True
#     elif 'context_text' in request.form and request.form['context_text'].strip():
#         document = request.form['context_text']
#         is_pdf = False
#     else:
#         return render_template("error.html", error="Please provide either a PDF file or text context."), 400
#     # Proceed with processing
#     try:
        
#         question = request.form.get("question")
#         if not question or not question.strip():
#             return render_template("error.html", error="Question is required."), 400
        
#         # Initialize InferenceClient
#         client = InferenceClient(
#             model="meta-llama/Llama-3.1-8B-Instruct",
#             token=token
#             # provider="hf-inference"
#         )
        
#         # Construct prompt
#         prompt = f"Context: {document}\nQuestion: {question}\nAnswer:"
        
#         # Query the model
#         answer = client.text_generation(
#             prompt,
#             max_new_tokens=512,
#             temperature=0.7,
#             top_p=0.9,
#             return_full_text=False
#         )
        
#         # Format response
#         response = {
#             "result": answer,
#             "source_documents": [{"page_content": document}] if is_pdf else []
#         }
        
#         return render_template(
#             "result.html",
#             result=response["result"],
#             sources=[doc["page_content"] for doc in response.get("source_documents", [])],
#             input_type="chat"
#         )
    
#     except Exception as e:
#         error_msg = str(e)
#         if "401" in error_msg or "403" in error_msg:
#             error_msg += " Authentication failed. Verify your token at https://huggingface.co/settings/tokens and ensure access to meta-llama/Llama-3.1-8B-Instruct at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct."
#         elif "404" in error_msg:
#             error_msg += " Model not found. Confirm the model ID or check Hugging Face Inference API status."
#         elif "Provider" in error_msg:
#             error_msg += " Check provider settings at https://hf.co/settings/inference-providers."
#         logger.error(f"Error in chat: {error_msg}\n{traceback.format_exc()}")
#         return render_template(
#             "error.html",
#             error=f"An error occurred: {error_msg}"
#         ), 500