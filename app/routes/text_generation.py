from flask import Blueprint, request, render_template
from ..models.text_generation import TextGenerator
from ..utils.validators import validate_text_input
from ..utils.logger import setup_logger

text_generation_bp = Blueprint("text_generation", __name__)
logger = setup_logger()

@text_generation_bp.route("/generate", methods=["POST", "GET"])
def generate():
    if request.method == "GET":
        return render_template("generate.html")
    try:
        prompt = request.form.get("prompt")
        if not validate_text_input(prompt):
            logger.error("Invalid prompt")
            return render_template("result.html", result="Invalid or empty prompt"), 400
        generator = TextGenerator()
        generated_text = generator.generate(prompt)
        logger.info("Text generated successfully")
        return render_template("result.html", result=generated_text)
    except Exception as e:
        logger.error(f"Error in generate: {e}")
        return render_template("result.html", result=f"Error: {e}"), 500