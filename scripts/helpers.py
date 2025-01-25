import os

from dotenv import load_dotenv


def setup_environment():
    """
    Set up the environment variables for the project.
    This function should be called at the beginning of the script.
    Warning: This function should be called before importing any huggingface library.
    """
    # Load the Hugging Face API key from .env file
    load_dotenv()

    # Update the cache directory of the transformers library
    models_dir = "models"
    os.environ["HF_HOME"] = models_dir
