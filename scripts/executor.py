import time
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

import dill as pickle
import torch
from transformers import pipeline

from scripts.dataset import RiddleQuestion

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PromptMessage:
    """Represents a single message in a model prompt with role and content"""

    role: str  # The role of the message sender (e.g. 'system', 'user', 'assistant')
    content: str  # The actual message content


@dataclass
class ModelExecutionResult:
    """Stores model execution results along with the prompts used"""

    model: str
    riddle_question: RiddleQuestion
    prompt_messages: list[dict]
    output: list[any]
    duration_in_seconds: int

    def model_name(self) -> str:
        """Returns the name of the model used"""
        return self.model

    def prompt_messages(self) -> list[dict]:
        """Returns the list of prompt messages used"""
        return self.prompt_messages

    def output(self) -> list[any]:
        """Returns the raw output from the model"""
        return self.output

    def duration_in_seconds(self) -> int:
        """Returns the duration of the model execution in seconds"""
        return self.duration_in_seconds

    def model_response(self) -> str:
        """Returns the final generated text content from the model"""
        return self.output[-1]["generated_text"][-1]["content"]


class ModelExecutor:
    """Handles execution of language models on riddle questions"""

    def __init__(
        self,
        models: list[str],
        dataset: list[RiddleQuestion],
        results_path: Path | str = Path("results"),
    ):
        self.models = models
        self.dataset = dataset
        self.results_path = Path(results_path)
        self._ensure_results_directory()
        logger.info(
            f"Initialized ModelExecutor with {len(models)} models and {len(dataset)} riddle questions"
        )

    def _ensure_results_directory(self):
        """Creates results directory if it doesn't exist"""
        if not self.results_path.exists():
            logger.warning(
                f"Results path {self.results_path} does not exist, creating it..."
            )
            self.results_path.mkdir(parents=True, exist_ok=True)

    def create_pipeline(self, model: str) -> pipeline:
        """Creates a text generation pipeline for the specified model"""
        logger.info(f"Creating pipeline for model: {model}")
        pipe = pipeline("text-generation", model)
        logger.info("Pipeline created successfully")
        return pipe

    def _cleanup_model_resources(self, pipe):
        """Cleans up model resources and frees memory"""
        logger.info("Cleaning up model resources...")
        del pipe
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Resource cleanup completed")

    def _save_results(self, results: dict, input_generator: Callable):
        """Saves execution results to disk"""
        logger.info("Saving final results to disk...")
        final_results_path = self.results_path / "all_results.pkl"
        results_metadata = {
            "models": self.models,
            "template": input_generator.__name__,
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results,
        }
        with open(final_results_path, "wb") as f:
            pickle.dump(results_metadata, f)
        logger.info(f"Final results saved to {final_results_path}")

    def _process_riddle(
        self,
        model: str,
        riddle_question: RiddleQuestion,
        pipe,
        input_generator: Callable,
    ) -> ModelExecutionResult:
        """Processes a single riddle question"""
        messages = input_generator(model, riddle_question)
        start_time = time.time()
        outputs = pipe(messages, max_new_tokens=256)
        duration_in_seconds = int(time.time() - start_time)
        logger.info(f"{model} execution took {duration_in_seconds} seconds")
        return ModelExecutionResult(
            model=model,
            riddle_question=riddle_question,
            prompt_messages=messages,
            output=outputs,
            duration_in_seconds=duration_in_seconds,
        )

    def run(
        self, input_generator: Callable[[str, RiddleQuestion], list[PromptMessage]]
    ) -> dict[str, list[ModelExecutionResult]]:
        """Runs the models on the dataset using the provided input generator"""
        logger.info("Starting model execution run")
        results: dict[str, list[ModelExecutionResult]] = {
            model: [] for model in self.models
        }

        for model in self.models:
            logger.info(f"Processing model: {model}")
            pipe = self.create_pipeline(model)
            try:
                model_results = []
                for i, riddle_question in enumerate(self.dataset, 1):
                    logger.debug(
                        f"Processing riddle {i}/{len(self.dataset)}: {riddle_question.id}"
                    )
                    result = self._process_riddle(
                        model, riddle_question, pipe, input_generator
                    )
                    model_results.append(result)
                results[model] = model_results
                logger.info(f"Completed processing for model: {model}")
            finally:
                self._cleanup_model_resources(pipe)

        self._save_results(results, input_generator)
        logger.info("Model execution run completed")
        return results
