# from flask import Blueprint, request, render_template
# from ..models.summarization import Summarizer
# from ..utils.validators import validate_text_input, validate_url
# from ..utils.file_handler import handle_pdf_upload
# from ..utils.logger import setup_logger

# summarization_bp = Blueprint("summarization", __name__)
# logger = setup_logger()

# @summarization_bp.route("/", methods=["POST"])
# def summarize():
#     try:
#         input_type = request.form.get("input_type")
#         summarizer = Summarizer()

#         if input_type == "text":
#             text = request.form.get("text")
#             if not validate_text_input(text):
#                 logger.error("Invalid text input")
#                 return render_template("result.html", result="Invalid or empty text"), 400
#             summary = summarizer.summarize_text(text)
#         elif input_type == "url":
#             url = request.form.get("url")
#             if not validate_url(url):
#                 logger.error("Invalid URL")
#                 return render_template("result.html", result="Invalid URL"), 400
#             summary = summarizer.summarize_url(url)
#         elif input_type == "pdf":
#             filepath = handle_pdf_upload(request, "pdf")
#             if isinstance(filepath, tuple):
#                 logger.error(f"PDF upload error: {filepath[0]}")
#                 return render_template("result.html", result=filepath[0]), 400
#             summary = summarizer.summarize_pdf(filepath)
#         else:
#             logger.error("Invalid input type")
#             return render_template("result.html", result="Invalid input type"), 400

#         logger.info(f"Summary generated for input_type: {input_type}")
#         return render_template("result.html", result=summary)
#     except Exception as e:
#         logger.error(f"Error in summarize: {e}")
#         return render_template("result.html", result=f"Error: {e}"), 500


from flask import Blueprint, request, render_template
from ..models.summarization import Summarizer
from ..utils.validators import validate_text_input, validate_url
from ..utils.file_handler import handle_pdf_upload
from ..utils.logger import setup_logger

summarization_bp = Blueprint("summarization", __name__)
logger = setup_logger()

@summarization_bp.route("/summarize", methods=["GET", "POST"])
def summarize():
    if request.method == "GET":
        return render_template("summarize.html")
    
    try:
        input_type = request.form.get("input_type")
        summarizer = Summarizer()

        if input_type == "text":
            text = request.form.get("text")
            if not validate_text_input(text):
                logger.error("Invalid text input")
                return render_template("result.html", result="Invalid or empty text"), 400
            summary = summarizer.summarize_text(text)
        elif input_type == "url":
            url = request.form.get("url")
            if not validate_url(url):
                logger.error("Invalid URL")
                return render_template("result.html", result="Invalid URL"), 400
            summary = summarizer.summarize_url(url)
        elif input_type == "pdf":
            filepath = handle_pdf_upload(request, "pdf")
            if isinstance(filepath, tuple):
                logger.error(f"PDF upload error: {filepath[0]}")
                return render_template("result.html", result=filepath[0]), 400
            summary = summarizer.summarize_pdf(filepath)
        else:
            logger.error("Invalid input type")
            return render_template("result.html", result="Invalid input type"), 400

        logger.info(f"Summary generated for input_type: {input_type}")
        return render_template("result.html", result=summary)
    except Exception as e:
        logger.error(f"Error in summarize: {e}")
        return render_template("result.html", result=f"Error: {e}"), 500