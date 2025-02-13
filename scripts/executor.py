import time
import logging
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

import dill as pickle
from tqdm.auto import tqdm
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

    def __init__(self, models: list[LLM], riddle_dataset: list[RiddleQuestion]):
        """Initialize executor with models and dataset"""
        self.models = models
        self.riddle_dataset = riddle_dataset
        logger.info(
            f"Initialized executor with {len(models)} models and {len(riddle_dataset)} riddles"
        )
        self.results_dir = Path("results")
        if not self.results_dir.exists():
            self.results_dir.mkdir()
            logger.info(f"Created results directory at {self.results_dir}")

        self.checkpoints_dir = self.results_dir / "checkpoints"
        if not self.checkpoints_dir.exists():
            self.checkpoints_dir.mkdir()
            logger.info(f"Created checkpoints directory at {self.checkpoints_dir}")

    def _execute_riddle(
        self,
        model: LLM,
        riddle: RiddleQuestion,
        prompt_template: ChatPromptTemplate,
        args_generator: Callable[[RiddleQuestion], dict],
    ) -> ExecutionResult:
        """Execute model on a single riddle"""
        template_args = args_generator(riddle)
        start_time = time.time()
        output = model.generate(prompt_template, template_args)
        delta = time.time() - start_time

        return ExecutionResult(
            model_name=model.name,
            riddle=riddle,
            prompt_used=template_args,
            model_output=output,
            execution_time=delta,
        )

    def execute(
        self,
        prompt_template: ChatPromptTemplate,
        args_generator: Callable[[RiddleQuestion], dict],
        dump_to_pickle: bool = False,
        file_name: str | None = None,
        create_checkpoints: bool = False,
    ) -> dict[str, list[ExecutionResult]]:
        """Execute all models on the dataset"""

        if file_name is None:
            file_name = "results.pkl"
            logger.warning(f"No file name provided, using default: {file_name}")

        logger.info("Starting execution")
        results: dict[str, list[ExecutionResult]] = {
            model.name: [] for model in self.models
        }

        for model in self.models:
            logger.info(f"Processing {model.name}")
            model.setup()

            try:
                model_results = []
                for riddle in tqdm(
                    self.riddle_dataset,
                    desc=model.name,
                    total=len(self.riddle_dataset),
                ):
                    result = self._execute_riddle(
                        model,
                        riddle,
                        prompt_template,
                        args_generator,
                    )
                    model_results.append(result)
                results[model.name] = model_results

                # Dump results to pickle checkpoint if checkpointing is enabled
                if create_checkpoints:
                    file_path = self.checkpoints_dir / f"{model.name}_{file_name}.pkl"
                    logger.info(f"Dumping results to {file_path}")
                    with open(file_path, "wb") as f:
                        pickle.dump(model_results, f)
            finally:
                model.cleanup()

        if dump_to_pickle:
            file_path = self.results_dir / file_name
            logger.info(f"Dumping results to {file_path}")
            with open(file_path, "wb") as f:
                pickle.dump(results, f)
        return results
