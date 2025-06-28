import logging
from app import create_app

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    logger.info("Starting Flask application")
    app = create_app()
    logger.info("Flask application created")
    if __name__ == "__main__":
        logger.info("Running Flask server")
        app.run(host="0.0.0.0", port=5000, debug=True)
except Exception as e:
    logger.error(f"Failed to start application: {e}")
    raise