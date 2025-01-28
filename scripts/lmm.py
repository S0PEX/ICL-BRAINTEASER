from collections.abc import Callable

from transformers import pipeline
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

"""
NOTE: These classes serve as temporary replacements for LangChain's LLM and HuggingFacePipeline classes
due to current tokenizer limitations in those implementations. The LangChain classes don't provide
straightforward ways to customize parsing or override the generate() method without significant
modifications, especially when trying to maintain compatibility with RAG (Retrieval Augmented Generation)
chains. This custom implementation allows for more control over the tokenization and generation for custom models such as Gemma that don't support the default system, user, and assistant roles.

This is intended to be replaced once LangChain's implementation provides more flexible customization
options or the tokenizer issues are resolved.
"""


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


class LLM:
    """Language model class that wraps HuggingFace models for text generation"""

    def __init__(self, model: str, device: int | None = None):
        """Initialize the language model with the specified model name and device

        Args:
            model (str): Name/path of the HuggingFace model to use
            device (int | None): Device ID to run the model on. None for CPU, int for specific GPU
        """
        self.model = model
        self.llm = pipeline(
            model=model,
            task="text-generation",
            device=device if device is not None else -1,
        )

    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        """Convert a LangChain message to a dict format for the model.

        Args:
            message (BaseMessage): Message to convert

        Returns:
            dict: Message in dict format with role and content
        """
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        elif isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    def _parse_generation(self, generation: list[dict]) -> ChatHistory:
        """Parse model generation output into a ChatHistory.

        Args:
            generation (list[dict]): Raw generation output from model

        Returns:
            ChatHistory: Parsed chat history
        """
        messages = []
        for gen in generation:
            for msg in gen["generated_text"]:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        return ChatHistory(messages)

    def generate(self, chat_template: ChatPromptTemplate, args: dict) -> ChatHistory:
        """Generate text based on the provided chat template and arguments.

        Args:
            chat_template (ChatPromptTemplate): Template for formatting the prompt
            args (dict): Arguments to fill in the template

        Returns:
            ChatHistory: Generated chat history
        """
        messages = [
            self._convert_message_to_dict(msg)
            for msg in chat_template.format_messages(**args)
        ]
        generations = self.llm(messages)
        return self._parse_generation(generations)

    def cleanup(self):
        """Cleanup the model by freeing up resources.

        This method should be called when the LLM instance is no longer needed
        to properly free system resources.
        """
        del self.llm

    def __or__(self, template: ChatPromptTemplate) -> Callable[[dict], ChatHistory]:
        """Overloads the | operator to create a bound template function.

        This allows using the | operator to bind a template to the LLM instance,
        creating a callback function that can generate responses using that template.

        Example:
            llm = LLM("model_name")
            template = ChatPromptTemplate(...)
            generate_response = llm | template
            history = generate_response({"arg1": "value1"})

        Args:
            template (ChatPromptTemplate): The chat prompt template to bind to this LLM

        Returns:
            Callable[[dict], ChatHistory]: A function that takes template arguments and
                returns the generated chat history
        """
        return lambda args: self.generate(template, args)

    @property
    def name(self) -> str:
        """Get the name of the model."""
        return self.model
