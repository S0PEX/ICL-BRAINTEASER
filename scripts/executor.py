import time
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

import dill as pickle
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue

from scripts.lmm import LLM
from scripts.dataset import RiddleQuestion

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a model on a single riddle question"""

    model_name: str
    riddle: RiddleQuestion
    prompt_used: PromptValue
    model_output: str
    execution_time: int

    def get_model_name(self) -> str:
        """Get the name of the model used"""
        return self.model_name

    def get_prompt(self) -> PromptValue:
        """Get the prompt that was used"""
        return self.prompt_used

    def get_output(self) -> str:
        """Get the raw model output"""
        return self.model_output

    def get_execution_time(self) -> int:
        """Get execution duration in seconds"""
        return self.execution_time

    def get_response(self) -> str:
        """Get the final model response"""
        return self.model_output


class Executor:
    """Executes language models on riddle questions"""

    def __init__(
        self,
        models: list[str],
        riddle_dataset: list[RiddleQuestion],
        output_dir: Path | str = Path("results"),
    ):
        """Initialize executor with models and dataset"""
        self.model_names = models
        self.riddle_dataset = riddle_dataset
        self.output_dir = Path(output_dir)
        self._init_output_dir()
        logger.info(
            f"Initialized executor with {len(models)} models and {len(riddle_dataset)} riddles"
        )

    def _init_output_dir(self):
        """Initialize output directory"""
        if not self.output_dir.exists():
            logger.warning(f"Creating output directory: {self.output_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _persist_results(self, results: dict, args_generator: Callable):
        """Save results to disk"""
        logger.info("Persisting results...")
        output_path = self.output_dir / "all_results.pkl"
        metadata = {
            "models": self.model_names,
            "template": args_generator.__name__,
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results,
        }
        with open(output_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Results saved to {output_path}")

    def _execute_riddle(
        self,
        model_name: str,
        riddle: RiddleQuestion,
        model: LLM,
        prompt_template: ChatPromptTemplate,
        args_generator: Callable[[RiddleQuestion], dict],
    ) -> ExecutionResult:
        """Execute model on a single riddle"""
        template_args = args_generator(riddle)
        start = time.time()
        output = model.generate(prompt_template, template_args)
        duration = int(time.time() - start)
        logger.info(f"{model_name} took {duration}s")
        return ExecutionResult(
            model_name=model_name,
            riddle=riddle,
            prompt_used=template_args,
            model_output=output,
            execution_time=duration,
        )

    def execute(
        self,
        prompt_template: ChatPromptTemplate,
        args_generator: Callable[[RiddleQuestion], dict],
    ) -> dict[str, list[ExecutionResult]]:
        """Execute all models on the dataset"""
        logger.info("Starting execution")
        results: dict[str, list[ExecutionResult]] = {
            model: [] for model in self.model_names
        }

        for model_name in self.model_names:
            logger.info(f"Processing {model_name}")
            model = LLM(model_name)
            try:
                model_results = []
                for i, riddle in enumerate(self.riddle_dataset, 1):
                    logger.debug(
                        f"Processing riddle {i}/{len(self.riddle_dataset)}: {riddle.id}"
                    )
                    result = self._execute_riddle(
                        model_name,
                        riddle,
                        model,
                        prompt_template,
                        args_generator,
                    )
                    model_results.append(result)
                results[model_name] = model_results
                logger.info(f"Completed {model_name}")
            finally:
                model.cleanup()

        self._persist_results(results, args_generator)
        logger.info("Execution completed")
        return results
