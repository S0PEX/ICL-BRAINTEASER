import time
import asyncio
import logging
from io import BufferedWriter
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

import dill as pickle
from tqdm.auto import tqdm
from more_itertools import batched
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue

from scripts.lmm import LLM, ChatHistory
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
    model_output: ChatHistory
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

    def get_response(self) -> ChatHistory:
        """Get the final model response"""
        return self.model_output


class Executor:
    """Executes language models on riddle questions"""

    def __init__(self, models: list[LLM]):
        """Initialize executor with models and dataset"""
        self.models = models
        logger.info(f"Initialized executor with {len(models)} models.")
        self.results_dir = Path("results")
        self.checkpoints_dir = self.results_dir / "checkpoints"

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

    async def _aexecute_riddle(
        self,
        model: LLM,
        riddle: RiddleQuestion,
        prompt_template: ChatPromptTemplate,
        args_generator: Callable[[RiddleQuestion], dict],
    ) -> ExecutionResult:
        """Execute model on a single riddle"""
        template_args = args_generator(riddle)
        start_time = time.time()
        output = await model.agenerate(prompt_template, template_args)
        delta = time.time() - start_time

        return ExecutionResult(
            model_name=model.name,
            riddle=riddle,
            prompt_used=template_args,
            model_output=output,
            execution_time=delta,
        )

    def _save_open_file(self, file_path: Path | str) -> BufferedWriter:
        """Open a file for writing, creating parent directories if necessary"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return open(file_path, "wb")

    def execute(
        self,
        dataset: list[RiddleQuestion],
        prompt_template: ChatPromptTemplate,
        args_generator: Callable[[RiddleQuestion], dict],
        dump_to_pickle: bool = False,
        result_file_name: str | None = None,
        create_checkpoints: bool = False,
    ) -> dict[str, list[ExecutionResult]]:
        """Execute all models on the dataset"""

        if result_file_name is None:
            result_file_name = "results"
            logger.warning(f"No file name provided, using default: {result_file_name}")
        if result_file_name.endswith(".pkl"):
            result_file_name = result_file_name[:-4]
            logger.warning(
                f"File name should not have extension, removing extension: {result_file_name}"
            )

        logger.info("Starting execution")
        results: dict[str, list[ExecutionResult]] = {
            model.name: [] for model in self.models
        }

        for model in self.models:
            logger.info(f"Processing {model.name}")
            model.load()

            try:
                model_results = []
                for riddle in tqdm(
                    dataset,
                    desc=model.name,
                    total=len(dataset),
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
                    file_path = (
                        self.checkpoints_dir
                        / result_file_name
                        / f"{model.name}_{result_file_name}.pkl"
                    )
                    logger.info(f"Creating checkpoint: {file_path}")

                    with self._save_open_file(file_path) as f:
                        pickle.dump(model_results, f)
            finally:
                logger.info(f"Cleaning up {model.name}")
                model.cleanup()

        logger.info("Execution complete")

        if dump_to_pickle:
            file_path = self.results_dir / f"{result_file_name}.pkl"
            logger.info(f"Dumping results to {file_path}")
            with self._save_open_file(file_path) as f:
                pickle.dump(results, f)
        return results

    async def aexecute(
        self,
        dataset: list[RiddleQuestion],
        prompt_template: ChatPromptTemplate,
        args_generator: Callable[[RiddleQuestion], dict],
        batch_size: int = 5,
        dump_to_pickle: bool = False,
        result_file_name: str | None = None,
        create_checkpoints: bool = False,
    ) -> dict[str, list[ExecutionResult]]:
        """Asynchronously execute all models on the dataset in batches."""

        if result_file_name is None:
            result_file_name = "results"
            logger.warning(f"No file name provided, using default: {result_file_name}")
        if result_file_name.endswith(".pkl"):
            result_file_name = result_file_name[:-4]
            logger.warning(
                f"File name should not have extension, removing extension: {result_file_name}"
            )

        logger.info("Starting asynchronous execution")
        results: dict[str, list[ExecutionResult]] = {
            model.name: [] for model in self.models
        }

        # Create batched dataset
        batched_dataset = list(batched(dataset, batch_size))
        total_batches = len(batched_dataset)

        logger.info(
            f"Split dataset of {len(dataset)} items into {total_batches} "
            f"batches of size {batch_size}"
        )

        for model in self.models:
            logger.info(f"Processing {model.name}")
            model.load()

            model_results = []
            try:
                with tqdm(total=len(dataset), desc=model.name) as progress_bar:
                    for chunk in batched_dataset:
                        tasks = [
                            self._aexecute_riddle(
                                model, riddle, prompt_template, args_generator
                            )
                            for riddle in chunk
                        ]
                        chunk_results = await asyncio.gather(*tasks)
                        model_results.extend(chunk_results)
                        progress_bar.update(len(chunk))

                results[model.name] = model_results

                if create_checkpoints:
                    file_path = (
                        self.checkpoints_dir
                        / result_file_name
                        / f"{model.name}_{result_file_name}.pkl"
                    )
                    logger.info(f"Creating checkpoint: {file_path}")
                    with self._save_open_file(file_path) as f:
                        pickle.dump(model_results, f)

            finally:
                logger.info(f"Cleaning up {model.name}")
                model.cleanup()

        logger.info("Asynchronous execution complete")

        if dump_to_pickle:
            file_path = self.results_dir / f"{result_file_name}.pkl"
            logger.info(f"Dumping results to {file_path}")
            with self._save_open_file(file_path) as f:
                pickle.dump(results, f)

        return results
