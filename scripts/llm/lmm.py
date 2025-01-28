from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline


class LLM:
    """Language model class that wraps HuggingFace models for text generation"""

    def __init__(self, model: str, device: int | None = None):
        """Initialize the language model with the specified model name and device

        Args:
            model (str): Name/path of the HuggingFace model to use
            device (int): Device ID to run the model on (default: 0 for first GPU)
        """
        self.model = model
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=model, task="text-generation", device=device
        )

    def generate(self, chat_template: ChatPromptTemplate, args: dict) -> str:
        """Generate text based on the provided chat template and arguments

        Args:
            chat_template (ChatPromptTemplate): Template for formatting the prompt
            args (dict): Arguments to fill in the template

        Returns:
            str: Generated text response
        """
        chain = chat_template | self.llm
        return chain.invoke(args)

    def cleanup(self):
        """Clean up model resources and free memory"""

    @property
    def name(self) -> str:
        """Get the name of the model"""
        return self.model
