import time
import asyncio
import logging
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

import dill as pickle
from tqdm.auto import tqdm
from more_itertools import batched
from langchain.prompts.chat import ChatPromptTemplate

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
    model_output: ChatHistory
    execution_time: float  # Changed to float for more precision

    def get_model_name(self) -> str:
        """Get the name of the model used"""
        return self.model_name

    def get_output(self) -> ChatHistory:  # Fixed return type
        """Get the raw model output"""
        return self.model_output

    def get_execution_time(self) -> float:  # Changed to float
        """Get execution duration in seconds"""
        return self.execution_time

    def get_response(self) -> ChatHistory:
        """Get the final model response"""
        return self.model_output


@dataclass
class Dataset:
    """Dataset containing riddle questions"""

    name: str
    riddles: list[RiddleQuestion]

    @property
    def size(self) -> int:
        """Return the number of riddles in the dataset"""
        return len(self.riddles)

    def __iter__(self):
        """Allow iteration over riddles"""
        return iter(self.riddles)


@dataclass
class WrappedResults:
    """Container for execution results with metadata"""

    run_name: str
    suffix: str
    results: dict[str, dict[str, list[ExecutionResult]]]


class Executor:
    """Executes language models on riddle questions"""

    def __init__(self, models: list[LLM]):
        self.models = {model.name: model for model in models}
        logger.info(f"Initialized executor with {len(models)} models.")
        self.results_dir = Path("results")

    async def _execute_riddle(
        self,
        model: LLM,
        riddle: RiddleQuestion,
        prompt_template: ChatPromptTemplate,
        args_generator: Callable[[RiddleQuestion], dict[str, Any]],
        is_async: bool = True,
    ) -> ExecutionResult:
        """Execute a single riddle with a model"""
        template_args = args_generator(riddle)
        start_time = time.time()

        if is_async:
            output = await model.agenerate(prompt_template, template_args)
        else:
            output = model.generate(prompt_template, template_args)

        execution_time = time.time() - start_time

        return ExecutionResult(
            model_name=model.name,
            riddle=riddle,
            model_output=output,
            execution_time=execution_time,
        )

    @staticmethod
    def _sanitize_str(s: str) -> str:
        """Sanitize a name for file system use"""
        return "-".join(
            filter(
                None,
                "".join(c if c.isalnum() else "-" for c in s).lower().split("-"),
            )
        )

    def _get_paths(
        self, run_name: str | None, file_name_suffix: str | None
    ) -> tuple[str, Path, Path]:
        """Get sanitized run name and relevant paths"""
        sanitized_name = self._sanitize_str(run_name or "default-run")
        run_dir = self.results_dir / sanitized_name
        return (
            sanitized_name,
            run_dir / "checkpoints",
            run_dir / f"{sanitized_name}_{file_name_suffix}_results.pkl",
        )

    async def _process_model(
        self,
        pbar: tqdm,
        model: LLM,
        dataset: Dataset,
        prompt_template: ChatPromptTemplate | Callable,
        args_generator: Callable,
        checkpoints_dir: Path,
        create_checkpoints: bool,
        resume_from_checkpoint: bool,
        batch_size: int = 4,
        file_name_suffix: str | None = None,
        is_async: bool = True,
    ) -> list[ExecutionResult]:
        """Process a single model with riddles from a dataset"""
        checkpoint_path = (
            checkpoints_dir
            / f"{dataset.name}_{model.name}{'_' + file_name_suffix if file_name_suffix else ''}.pkl"
        )

        # Try to load from checkpoint first if enabled
        if resume_from_checkpoint and checkpoint_path.exists():
            with open(checkpoint_path, "rb") as f:
                logger.debug(f"Loading cached results from {checkpoint_path}")
                loaded_results = pickle.load(f)
                riddles_len = len(dataset.riddles)
                pbar.update(riddles_len)
                pbar.set_postfix(
                    {
                        "Dataset": dataset.name,
                        "Model": model.name,
                        "Progress": f"{riddles_len}/{riddles_len}",
                    }
                )
                return loaded_results

        model.load()
        riddles = dataset.riddles
        try:
            # Prepare template - handle callable template case
            template = (
                prompt_template(model.name)
                if callable(prompt_template)
                else prompt_template
            )

            results = []
            riddles_len = len(riddles)

            if is_async:
                # Process in batches
                batches = list(batched(riddles, batch_size))
                for i, chunk in enumerate(batches):
                    tasks = [
                        self._execute_riddle(
                            model, riddle, template, args_generator, True
                        )
                        for riddle in chunk
                    ]
                    chunk_results = await asyncio.gather(*tasks)
                    results.extend(chunk_results)

                    # Update progress bar
                    pbar.set_postfix(
                        {
                            "Dataset": dataset.name,
                            "Model": model.name,
                            "Progress": f"{min((i + 1) * batch_size, riddles_len)}/{riddles_len}",
                        }
                    )
                    pbar.update(len(chunk))
            else:
                # Process sequentially
                for i, riddle in enumerate(riddles):
                    result = await self._execute_riddle(
                        model, riddle, template, args_generator, False
                    )
                    results.append(result)

                    # Update progress bar
                    pbar.set_postfix(
                        {
                            "Dataset": dataset.name,
                            "Model": model.name,
                            "Progress": f"{i + 1}/{riddles_len}",
                        }
                    )
                    pbar.update(1)

            # Save checkpoint if enabled
            if create_checkpoints:
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(results, f)

            return results
        finally:
            model.cleanup()

    async def execute(
        self,
        input_data: Dataset | list[Dataset],
        prompt_template: ChatPromptTemplate | Callable,
        args_generator: Callable,
        run_name: str | None = None,
        file_name_suffix: str | None = None,
        dump_to_pickle: bool = False,
        create_checkpoints: bool = False,
        resume_from_checkpoint: bool = False,
        batch_size: int = 4,
    ) -> WrappedResults:
        """Synchronous execution"""
        return await self._execute_base(
            input_data,
            prompt_template,
            args_generator,
            run_name,
            file_name_suffix,
            dump_to_pickle,
            create_checkpoints,
            resume_from_checkpoint,
            batch_size,
            False,
        )

    async def aexecute(
        self,
        input_data: Dataset | list[Dataset],
        prompt_template: ChatPromptTemplate | Callable,
        args_generator: Callable,
        run_name: str | None = None,
        file_name_suffix: str | None = None,
        dump_to_pickle: bool = False,
        create_checkpoints: bool = False,
        resume_from_checkpoint: bool = False,
        batch_size: int = 4,
    ) -> WrappedResults:
        """Asynchronous execution"""
        return await self._execute_base(
            input_data,
            prompt_template,
            args_generator,
            run_name,
            file_name_suffix,
            dump_to_pickle,
            create_checkpoints,
            resume_from_checkpoint,
            batch_size,
            True,
        )

    async def _execute_base(
        self,
        input_data: Dataset | list[Dataset],
        prompt_template: ChatPromptTemplate | Callable,
        args_generator: Callable,
        run_name: str | None = None,
        file_name_suffix: str | None = None,
        dump_to_pickle: bool = False,
        create_checkpoints: bool = False,
        resume_from_checkpoint: bool = False,
        batch_size: int = 4,
        is_async: bool = True,
    ) -> WrappedResults:
        """Base execution logic for both sync and async operations"""
        # Setup paths and check for cached results
        sanitized_run_name = self._sanitize_str(run_name or "default-run")
        sanitized_suffix = (
            self._sanitize_str(file_name_suffix) if file_name_suffix else ""
        )
        run_name, checkpoints_dir, results_path = self._get_paths(
            sanitized_run_name, sanitized_suffix
        )

        # Ensure input is a list of datasets
        datasets = [input_data] if not isinstance(input_data, list) else input_data
        total_riddles = sum(
            len(dataset.riddles) * len(self.models) for dataset in datasets
        )

        # Log execution information
        suffix_info = f" with suffix '{file_name_suffix}'" if file_name_suffix else ""
        logger.info(
            f"Starting execution '{run_name}{suffix_info}': "
            f"{len(datasets)} dataset(s) x {len(self.models)} model(s) = "
            f"{total_riddles} riddle evaluations"
        )

        # Initialize progress bar
        desc = f"{run_name}({sanitized_suffix})" if sanitized_suffix else run_name
        with tqdm(total=total_riddles, desc=desc) as pbar:
            # Try loading complete results if enabled
            if resume_from_checkpoint and results_path.exists():
                logger.debug(f"Loading cached results from {results_path}")
                with open(results_path, "rb") as f:
                    wrapped_results = pickle.load(f)
                pbar.update(total_riddles)
                pbar.set_postfix({"Status": f"Loaded from cache: {results_path.name}"})
                return wrapped_results

            # Process all models and datasets
            results = {}
            for dataset in datasets:
                logger.debug(
                    f"Processing dataset: {dataset.name} with {dataset.size} riddles"
                )
                dataset_results = {}

                for model_name, model in self.models.items():
                    dataset_results[model_name] = await self._process_model(
                        pbar,
                        model,
                        dataset,
                        prompt_template,
                        args_generator,
                        checkpoints_dir,
                        create_checkpoints,
                        resume_from_checkpoint,
                        batch_size,
                        sanitized_suffix,
                        is_async,
                    )

                results[dataset.name] = dataset_results

            # Wrap results
            wrapped_results = WrappedResults(run_name, sanitized_suffix, results)

            # Save results if needed
            if dump_to_pickle:
                results_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving results to {results_path}")
                with open(results_path, "wb") as f:
                    pickle.dump(wrapped_results, f)

            return wrapped_results
