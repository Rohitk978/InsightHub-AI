import os
from flask import Flask
from .config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Register blueprints lazily
    from .routes.home import home_bp
    from .routes.summarization import summarization_bp
    from .routes.text_generation import text_generation_bp
    from .routes.chatbot import chatbot_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(summarization_bp)
    app.register_blueprint(text_generation_bp)
    app.register_blueprint(chatbot_bp)

    return app