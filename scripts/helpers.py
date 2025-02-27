import os

from dotenv import load_dotenv


def setup_environment() -> None:
    """
    Set up the environment variables for the project.

    This function loads environment variables from a .env file and configures
    the Hugging Face cache directory to use a local 'models' folder.

    Returns:
        None: This function doesn't return any value.

    Note:
        This function should be called at the beginning of the script,
        before importing any huggingface library.
    """
    # Load the Hugging Face API key from .env file
    load_dotenv()

    # Update the cache directory of the transformers library
    models_dir: str = "models"
    os.environ["HF_HOME"] = models_dir
