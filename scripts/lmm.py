import logging
from abc import ABC, abstractmethod

import ollama
from langchain_ollama import ChatOllama
from langchain_openai import OpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage

"""
NOTE: These classes serve as temporary replacements for LangChain's LLM and HuggingFacePipeline classes
due to current tokenizer limitations in those implementations. The LangChain classes don't provide
straightforward ways to customize parsing or override the generate() method without significant
modifications, especially when trying to maintain compatibility with RAG (Retrieval Augmented Generation)
chains. This custom implementation allows for more control over the tokenization and generation for custom models such as Gemma that don't support the default system, user, and assistant roles.

This is intended to be replaced once LangChain's implementation provides more flexible customization
options or the tokenizer issues are resolved.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress third-party Langchain HTTP logging (https://github.com/langchain-ai/langchain/issues/14065#issuecomment-2252540350)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class ChatHistory:
    """A class representing a chat conversation history containing a list of messages."""

    def __init__(self, messages: list[BaseMessage]):
        """Initialize a ChatHistory with a list of messages.

        Args:
            messages (list[BaseMessage]): List of BaseMessage objects representing the chat history
        """
        self.messages = messages

    def __iter__(self):
        """Make ChatHistory iterable by returning an iterator over the messages.

        Returns:
            iterator: Iterator over the messages in the chat history
        """
        return iter(self.messages)

    def __str__(self) -> str:
        """Convert the chat history to a readable string format.

        Returns:
            str: String representation of chat history with each message on a new line
                 in "role: content" format, where role is lowercase without "Message" suffix
        """
        output = []
        for message in self.messages:
            role = message.__class__.__name__.replace("Message", "")
            content = message.content
            output.append(f"{role}: {content}")
        return "\n".join(output)

    def get_ai_response(self) -> AIMessage:
        """Get the last AI message from the chat history.

        Returns:
            AIMessage: The last AI message in the chat history
        """
        return self.messages[-1]


class LLM(ABC):
    """Interface for language model implementations"""

    @abstractmethod
    def load():
        """Setup the model and any required resources."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, chat_template: ChatPromptTemplate, args: dict) -> ChatHistory:
        """Generate text based on the provided chat template and arguments.

        Args:
            chat_template (ChatPromptTemplate): Template for formatting the prompt
            args (dict): Arguments to fill in the template

        Returns:
            ChatHistory: Generated chat history
        """
        raise NotImplementedError

    async def agenerate(
        self, chat_template: ChatPromptTemplate, args: dict
    ) -> ChatHistory:
        """Generate text based on the provided chat template and arguments.

        Args:
            chat_template (ChatPromptTemplate): Template for formatting the prompt
            args (dict): Arguments to fill in the template

        Returns:
            ChatHistory: Generated chat history
        """
        raise NotImplementedError

    @abstractmethod
    def cleanup(self):
        """Cleanup the model by freeing up resources.

        This method should be called when the LLM instance is no longer needed
        to properly free system resources.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the model."""
        raise NotImplementedError


class vLLMModel(LLM):
    """Language model class that wraps Ollama models for text generation"""

    def __init__(self, model: str):
        """Initialize the language model with the specified model name and device

        Args:
            model (str): Name/path of the Ollama model to use
        """
        self.model = model
        self.llm = OpenAI(
            model=model,
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )

    def load(self):
        """Ensure that the model is pulled from ollama and ready for use."""

    def generate(self, chat_template: ChatPromptTemplate, args: dict) -> ChatHistory:
        """Generate text based on the provided chat template and arguments."""

        messages = chat_template.format_messages(**args)
        response = self.llm.invoke(messages)
        return ChatHistory([messages, response])

    async def agenerate(
        self, chat_template: ChatPromptTemplate, args: dict
    ) -> ChatHistory:
        """Generate text based on the provided chat template and arguments."""

        messages = chat_template.format_messages(**args)
        response = await self.llm.ainvoke(messages)
        return ChatHistory([messages, response])

    def cleanup(self):
        """Cleanup the model by freeing up resources."""

        # Delete image
        logger.info(
            "Ollama models will be deleted on demand and therefore this step is skipped!"
        )

    @property
    def name(self) -> str:
        """Get the name of the model."""

        return self.model


class OllamaModelBuilder:
    """Builder class for creating OllamaModel instances"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize the builder with base URL.

        Args:
            base_url (str): Base URL for the Ollama server
        """
        self.base_url = base_url

    def build_model(self, model_name: str) -> LLM:
        """Build an OllamaModel instance with the specified model name.

        Args:
            model_name (str): Name of the Ollama model to use

        Returns:
            LLM: Configured OllamaModel instance
        """
        return OllamaModel(model=model_name, base_url=self.base_url)


class OllamaModel(LLM):
    """Language model class that wraps Ollama models for text generation"""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        """Initialize the language model with the specified model name and device

        Args:
            model (str): Name/path of the Ollama model to use
        """
        self.model = model
        self.base_url = base_url
        self.llm = ChatOllama(model=model, base_url=base_url)
        self.ollama_client = ollama.Client(base_url)

    def load(self):
        """Ensure that the model is pulled from ollama and ready for use."""

        # Pull image
        logger.info(f"Pulling Ollama model: {self.model}")
        try:
            self.ollama_client.pull(self.model)
        except Exception as e:
            logger.error(f"Error pulling Ollama model: {e}")
            if "no space left on device" in str(e):
                logger.info("Deleting all ollama models to free up space")
                list_response = self.ollama_client.list()
                models = list_response.models
                for model in models:
                    self.ollama_client.delete(model.model)
                self.ollama_client.pull(self.model)
            else:
                raise e

    def generate(self, chat_template: ChatPromptTemplate, args: dict) -> ChatHistory:
        """Generate text based on the provided chat template and arguments."""

        messages = chat_template.format_messages(**args)
        response = self.llm.invoke(messages)
        return ChatHistory([messages, response])

    async def agenerate(
        self, chat_template: ChatPromptTemplate, args: dict
    ) -> ChatHistory:
        """Generate text based on the provided chat template and arguments."""

        messages = chat_template.format_messages(**args)
        response = await self.llm.ainvoke(messages)
        return ChatHistory([messages, response])

    def cleanup(self):
        """Cleanup the model by freeing up resources."""

        # Delete image
        logger.info(
            "Ollama models will be deleted on demand and therefore this step is skipped!"
        )

    @property
    def name(self) -> str:
        """Get the name of the model."""

        return self.model


# class HuggingfacePipelineLlm(LLM):
#     """Language model class that wraps HuggingFace models for text generation"""

#     def __init__(self, model: str, device: int | None = None):
#         """Initialize the language model with the specified model name and device

#         Args:
#             model (str): Name/path of the HuggingFace model to use
#             device (int | None): Device ID to run the model on. None for CPU, int for specific GPU
#         """
#         self.model = model
#         self.llm = pipeline(
#             model=model,
#             task="text-generation",
#             device=device if device is not None else -1,
#         )

#     def _convert_message_to_dict(self, message: BaseMessage) -> dict:
#         """Convert a LangChain message to a dict format for the model.

#         Args:
#             message (BaseMessage): Message to convert

#         Returns:
#             dict: Message in dict format with role and content
#         """
#         if isinstance(message, SystemMessage):
#             return {"role": "system", "content": message.content}
#         elif isinstance(message, HumanMessage):
#             return {"role": "user", "content": message.content}
#         else:
#             raise ValueError(f"Unsupported message type: {type(message)}")

#     def _parse_generation(self, generation: list[dict]) -> ChatHistory:
#         """Parse model generation output into a ChatHistory.

#         Args:
#             generation (list[dict]): Raw generation output from model

#         Returns:
#             ChatHistory: Parsed chat history
#         """
#         messages = []
#         for gen in generation:
#             for msg in gen["generated_text"]:
#                 role = msg["role"]
#                 content = msg["content"]
#                 if role == "system":
#                     messages.append(SystemMessage(content=content))
#                 elif role == "user":
#                     messages.append(HumanMessage(content=content))
#                 elif role == "assistant":
#                     messages.append(AIMessage(content=content))
#         return ChatHistory(messages)

#     def generate(self, chat_template: ChatPromptTemplate, args: dict) -> ChatHistory:
#         """Generate text based on the provided chat template and arguments.

#         Args:
#             chat_template (ChatPromptTemplate): Template for formatting the prompt
#             args (dict): Arguments to fill in the template

#         Returns:
#             ChatHistory: Generated chat history
#         """
#         messages = [
#             self._convert_message_to_dict(msg)
#             for msg in chat_template.format_messages(**args)
#         ]
#         generations = self.llm(messages)
#         return self._parse_generation(generations)

#     def cleanup(self):
#         """Cleanup the model by freeing up resources.

#         This method should be called when the LLM instance is no longer needed
#         to properly free system resources.
#         """
#         del self.llm

#     @property
#     def name(self) -> str:
#         """Get the name of the model."""
#         return self.model
